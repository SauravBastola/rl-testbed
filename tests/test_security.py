"""The local server must not leak files or accept requests from other sites.

Traversal paths are percent-encoded (``..%2F``): HTTP clients normalise a plain
``../`` before sending, so an unencoded test never reaches the server's route.
"""

import pytest
from fastapi.testclient import TestClient

SECRET = "top-secret-contents"


@pytest.fixture
def site(tmp_path):
    """A fake built frontend plus a secret file next to (outside) it."""
    dist = tmp_path / "srv" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "favicon.png").write_bytes(b"png")
    (tmp_path / "srv" / "secret.txt").write_text(SECRET)
    (tmp_path / "secret.txt").write_text(SECRET)
    return dist


def _app(dist, **kw):
    from rltestbed.server.app import create_app

    return create_app(web_dist=dist, **kw)


@pytest.mark.parametrize("path", [
    "/..%2Fsecret.txt",
    "/..%2F..%2Fsecret.txt",
    "/%2E%2E%2Fsecret.txt",
    "/assets%2F..%2F..%2Fsecret.txt",
])
def test_spa_fallback_cannot_escape_dist(site, path):
    r = TestClient(_app(site)).get(path)
    assert SECRET not in r.text
    # the SPA route answers with the app shell; the /assets mount with a 404
    assert r.text == "<html>app</html>" or r.status_code == 404


def test_spa_still_serves_files_inside_dist(site):
    c = TestClient(_app(site))
    assert c.get("/favicon.png").content == b"png"
    assert c.get("/lab/foundations/runs").text == "<html>app</html>"


def test_unknown_api_paths_are_404_not_the_app_shell(site):
    assert TestClient(_app(site)).get("/api/does-not-exist").status_code == 404


def test_no_cross_origin_access_by_default(site):
    c = TestClient(_app(site))
    r = c.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    pre = c.options("/api/runs", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "DELETE"})
    assert "access-control-allow-origin" not in pre.headers


def test_cors_can_be_enabled_explicitly(site):
    c = TestClient(_app(site, cors_origins=["http://localhost:5173"]))
    r = c.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_dns_rebinding_hosts_are_rejected(site):
    """A malicious domain rebound to 127.0.0.1 still sends its own Host header."""
    assert TestClient(_app(site), base_url="http://evil.example").get("/api/health").status_code == 400
    assert TestClient(_app(site), base_url="http://localhost:8000").get("/api/health").status_code == 200
    assert TestClient(_app(site), base_url="http://127.0.0.1:8000").get("/api/health").status_code == 200


def test_network_serving_can_accept_any_host(site):
    c = TestClient(_app(site, allowed_hosts=["*"]), base_url="http://192.168.1.20:8000")
    assert c.get("/api/health").status_code == 200
