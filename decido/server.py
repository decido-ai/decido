"""Jev-compatible HTTP server for Decido (standard library only).

    python -m decido.server --gguf decido-26b-a4b-q8_0.gguf --bundle bundle [--port 8080] [--threads 10] [--workers 1]

Endpoints (wire format of TypeSafe's Jev "System One" API):
    POST /v1/systemone   {"model"?, "state", "questions": {id: {type, instructions, criteria?}}}
                         -> {"model", "answers": {id: answer}, "usage": {"input_tokens", "output_tokens"}}
    GET  /v1/models      -> {"models": [{"name", "description", "release_date"}]}
    GET  /health         -> {"status": "ok"}

Existing Jev clients work by pointing them at this server, e.g. with the official SDK:
    TYPESAFE_BASE_URL=http://127.0.0.1:8080 TYPESAFE_API_KEY=local python your_app.py

Request rules follow the Jev API reference: bodies up to 32 MiB (413 above), optional gzip / x-gzip / zstd
Content-Encoding (415 for anything else or more than one layer; zstd needs Python 3.14+ or the `zstandard` package),
UTF-8 without BOM, finite numbers only, nesting up to 64 levels (400 otherwise), validation errors as 422 with a
FastAPI-style {"detail": [...]} body, and 400 `max_tokens_exceeded` for over-long requests. Supported question types:
noul, choice (2-255 options), score (2-10 levels). `box` and image assets are not supported (422).
Authentication: none by default (binds to 127.0.0.1); set DECIDO_API_KEY to require `Authorization: Bearer <key>`.
"""

from __future__ import annotations

import argparse
import gzip
import hmac
import json
import os
import queue
import sys
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from decido.predictor_llamacpp import DecidoLlamaCpp

MAX_BODY = 32 * 1024 * 1024
MAX_DEPTH = 64
MODEL_NAME = "decido-26b-a4b-q8_0"
MODELS = {"models": [{"name": MODEL_NAME, "description": "Decido 26B-A4B (Q8_0 GGUF, CPU): calibrated noul / choice / "
                      "score decisions; Jev-compatible wire format", "release_date": "2026-10-21"}]}


class RequestError(Exception):
    def __init__(self, status: int, body: dict):
        super().__init__(body)
        self.status, self.body = status, body


def _bad(msg: str, status: int = 400):
    return RequestError(status, {"detail": msg})


def _invalid(loc: list, msg: str, typ: str = "value_error", inp=None):
    return RequestError(422, {"detail": [{"type": typ, "loc": ["body", *loc], "msg": msg, "input": inp}]})


# ----------------------------------------------------------------------------- body decoding

def _zstd_decompress(data: bytes) -> bytes:
    try:
        from compression import zstd  # Python 3.14+
        return zstd.decompress(data)
    except ImportError:
        pass
    try:
        import zstandard
    except ImportError as e:
        raise _bad("zstd Content-Encoding needs Python 3.14+ or `pip install zstandard`", 415) from e
    out = zstandard.ZstdDecompressor().stream_reader(data).read(MAX_BODY + 1)
    return out


def decode_body(raw: bytes, encoding: str | None) -> dict:
    enc = [e.strip().lower() for e in (encoding or "").split(",") if e.strip() and e.strip().lower() != "identity"]
    if len(enc) > 1:
        raise _bad("only one Content-Encoding layer is supported", 415)
    if enc:
        if enc[0] in ("gzip", "x-gzip"):
            try:
                raw = gzip.decompress(raw)
            except (OSError, EOFError, zlib.error) as e:
                raise _bad(f"malformed gzip body: {e}") from e
        elif enc[0] == "zstd":
            try:
                raw = _zstd_decompress(raw)
            except RequestError:
                raise
            except Exception as e:  # noqa: BLE001 - any decoder error is a malformed body
                raise _bad(f"malformed zstd body: {e}") from e
        else:
            raise _bad(f"unsupported Content-Encoding {enc[0]!r} (use gzip, x-gzip or zstd)", 415)
        if len(raw) > MAX_BODY:
            raise _bad("decompressed body is over 32 MiB", 413)
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _bad("body must be UTF-8 without a byte-order mark")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _bad(f"body is not valid UTF-8: {e}") from e

    def no_constant(c):
        raise ValueError(f"non-finite number {c} is not allowed")

    def finite_float(s):
        v = float(s)
        if v in (float("inf"), float("-inf")):
            raise ValueError(f"number {s} is too large to represent")
        return v
    try:
        body = json.loads(text, parse_constant=no_constant, parse_float=finite_float)
    except ValueError as e:
        raise _bad(f"invalid JSON: {e}") from e
    if _depth(body) > MAX_DEPTH:
        raise _bad(f"nesting deeper than {MAX_DEPTH} levels")
    return body


def _depth(o, d=0):
    if isinstance(o, dict):
        return max([_depth(v, d + 1) for v in o.values()] or [d + 1])
    if isinstance(o, list):
        return max([_depth(v, d + 1) for v in o] or [d + 1])
    return d


# ----------------------------------------------------------------------------- validation

def validate(body) -> dict:
    if not isinstance(body, dict):
        raise _invalid([], "request body must be a JSON object", "model_attributes_type", body)
    if "state" not in body:
        raise _invalid(["state"], "Field required", "missing")
    if body.get("assets"):
        raise _invalid(["assets"], "assets (images) are not supported by Decido", "value_error", None)
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise _invalid(["questions"], "questions must be a non-empty object of id -> question", "value_error", qs)
    for qid, q in qs.items():
        loc = ["questions", qid]
        if not isinstance(q, dict):
            raise _invalid(loc, "question must be an object", "model_attributes_type", q)
        t = q.get("type")
        if t == "box":
            raise _invalid(loc + ["type"], "box questions (image regions) are not supported by Decido", "value_error", t)
        if t not in ("noul", "choice", "score"):
            raise _invalid(loc + ["type"], "type must be one of 'noul', 'choice', 'score'", "literal_error", t)
        if q.get("instructions") in (None, "", [], {}):
            raise _invalid(loc + ["instructions"], "instructions are required", "missing")
        c = q.get("criteria")
        if t == "noul" and c is not None and not (isinstance(c, dict) and set(c) <= {"true", "false"}):
            raise _invalid(loc + ["criteria"], "noul criteria must be an object with optional 'true' / 'false' keys",
                           "value_error", c)
        if t == "choice" and not (isinstance(c, dict) and 2 <= len(c) <= 255):
            raise _invalid(loc + ["criteria"], "choice criteria must be an object with 2-255 options", "value_error", c)
        if t == "score" and not (isinstance(c, list) and 2 <= len(c) <= 10):
            raise _invalid(loc + ["criteria"], "score criteria must be a list of 2-10 levels", "value_error", c)
    return {"state": body["state"], "questions": qs}


# ----------------------------------------------------------------------------- server

class Pool:
    """One vserve process per worker; requests take a free worker (each worker serves one request at a time)."""

    def __init__(self, model: DecidoLlamaCpp):
        self.model = model
        self.free = queue.Queue()
        for p in model.procs:
            self.free.put(p)

    def predict(self, request: dict) -> dict:
        proc = self.free.get()
        try:
            return self.model._predict(proc, request)
        finally:
            self.free.put(proc)


def make_handler(pool: Pool, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "decido"

        def log_message(self, fmt, *args):  # one compact line per request on stderr
            sys.stderr.write(f"{self.address_string()} {self.command} {self.path} {fmt % args}\n")

        def _send(self, status: int, body: dict, rid: str, extra: dict | None = None):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("x-typesafe-request-id", rid)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _auth_ok(self) -> bool:
            if not api_key:
                return True
            scheme, _, token = self.headers.get("Authorization", "").strip().partition(" ")
            return scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), api_key)

        def do_GET(self):
            rid = uuid.uuid4().hex
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/health":
                return self._send(200, {"status": "ok", "model": MODEL_NAME}, rid)
            if not self._auth_ok():
                return self._send(401, {"detail": "missing or invalid API key"}, rid)
            if path == "/v1/models":
                return self._send(200, MODELS, rid)
            return self._send(404, {"detail": "not found"}, rid)

        def do_POST(self):
            rid = uuid.uuid4().hex
            t0 = time.perf_counter()
            path = self.path.split("?", 1)[0].rstrip("/")
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0:
                return self._send(400, {"detail": "invalid Content-Length"}, rid)
            if n > MAX_BODY:
                self.close_connection = True
                return self._send(413, {"detail": "body is over 32 MiB"}, rid)
            raw = self.rfile.read(n) if n else b""
            if path != "/v1/systemone":
                return self._send(404, {"detail": "not found"}, rid)
            if not self._auth_ok():
                return self._send(401, {"detail": "missing or invalid API key"}, rid)
            try:
                request = validate(decode_body(raw, self.headers.get("Content-Encoding")))
                out = pool.predict(request)
            except RequestError as e:
                return self._send(e.status, e.body, rid)
            except ValueError as e:  # raised by the readout's own validation
                msg = str(e)
                if msg.startswith("max_tokens_exceeded"):
                    return self._send(400, {"detail": msg}, rid)
                return self._send(422, {"detail": [{"type": "value_error", "loc": ["body"], "msg": msg, "input": None}]},
                                  rid)
            except Exception as e:  # noqa: BLE001 - never crash the server on one request
                return self._send(500, {"detail": f"internal error: {type(e).__name__}: {e}"}, rid)
            out["model"] = MODEL_NAME
            ms = (time.perf_counter() - t0) * 1000
            return self._send(200, out, rid, {"x-decido-latency-ms": f"{ms:.0f}"})

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description="Jev-compatible HTTP server for Decido (CPU).")
    ap.add_argument("--gguf", default="decido-26b-a4b-q8_0.gguf")
    ap.add_argument("--bundle", default="bundle")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--threads", type=int, default=10, help="CPU threads per worker")
    ap.add_argument("--workers", type=int, default=1, help="parallel model processes (~4 GB extra RAM each)")
    ap.add_argument("--perm", type=int, default=1, help="option orders averaged for choice questions (4 = less order-sensitive)")
    a = ap.parse_args(argv)
    api_key = os.environ.get("DECIDO_API_KEY") or None
    if a.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print("warning: listening on a non-local address without DECIDO_API_KEY; anyone who can reach this port can "
              "use the model", file=sys.stderr)
    print(f"loading {a.gguf} ({a.workers} worker(s) x {a.threads} threads) ...", file=sys.stderr, flush=True)
    model = DecidoLlamaCpp(gguf=a.gguf, bundle=a.bundle, threads=a.threads, workers=a.workers, perm=a.perm,
                           jev_wire=True)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(Pool(model), api_key))
    print(f"Decido is serving the Jev-compatible API on http://{a.host}:{a.port}/v1/systemone", file=sys.stderr,
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        model.close()


if __name__ == "__main__":
    main()
