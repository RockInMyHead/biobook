import contextlib
import subprocess
from unittest import mock
import importlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from docx import Document
from fastapi.testclient import TestClient
from PIL import Image
from pptx import Presentation


class AuditRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_dir = Path.cwd()
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        source = Path(__file__).resolve().parents[1]
        for name in ("static", "templates", "content"):
            shutil.copytree(source / name, root / name)
        cls.addClassCleanup(cls.temp.cleanup)
        os.chdir(root)
        cls.addClassCleanup(os.chdir, cls.previous_dir)
        sys.path.insert(0, str(source))
        settings = {"BASE_PATH": "/bio-notebook", "NOTEBOOKS_PATH": str(root / "notebooks"),
                    "EXPERIMENT_FILES_DIR": str(root / "attachments"),
                    "ADMIN_KEY": "local-test-admin-config-only", "TG_BOT_TOKEN": "", "COOKIE_SECURE": "true"}
        cls.previous_env = {key: os.environ.get(key) for key in settings}
        os.environ.update(settings)
        cls.addClassCleanup(cls._restore_env)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.m = importlib.import_module("app")
        cls.addClassCleanup(cls.m.engine.dispose)
        cls.prefix = "/bio-notebook"
        cls.password = "Local-test-password-2027"
        with cls.m.SessionLocal() as db:
            for user in ("owner", "member", "observer"):
                db.add(cls.m.UserDB(username=user, password=cls.m.hash_password(cls.password),
                                    email=user + "@example.com", plan="pro"))
            db.commit()

    @classmethod
    def _restore_env(cls):
        for key, val in cls.previous_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def request(self, user=None):
        client = TestClient(self.m.app, base_url="https://testserver")
        self.addCleanup(client.close)
        if user:
            response = client.post(self.prefix + "/login", data={"username": user, "password": self.password})
            self.assertEqual(200, response.status_code)
        return client

    def admin(self):
        client = self.request()
        response = client.post(self.prefix + self.m.LAB_ADMIN_PATH, data={"admin_key": self.m.ADMIN_KEY},
                               follow_redirects=False)
        self.assertEqual(302, response.status_code)
        self.assertNotIn("?", response.headers["location"])
        self.assertIn(self.m.ADMIN_COOKIE, client.cookies)
        return client

    def experiment(self, owner):
        result = owner.post(self.prefix + "/create_experiment", data={"user": "owner", "name": "QA sample",
                            "description": "Local fixture", "template": "blank"}, follow_redirects=False)
        self.assertEqual(302, result.status_code)
        route = urlsplit(result.headers["location"]).path
        exp, node = route.split("/")[-3], route.split("/")[-1]
        self.assertEqual(200, owner.get(route).status_code)
        return exp, node, route

    def test_prefix_serves_styles(self):
        client = self.request()
        for filename in ("styles.css", "tailwind.css"):
            response = client.get(self.prefix + "/static/" + filename)
            self.assertEqual(200, response.status_code)
            self.assertIn("text/css", response.headers["content-type"])

    def test_note_versions_and_read_only_export(self):
        owner = self.request("owner")
        exp, node, route = self.experiment(owner)
        params = {"exp_id": exp, "node_id": node}
        note_save = self.prefix + "/api/node/save"
        first = owner.post(note_save, json={**params, "html": "<p>Archived draft</p>", "revision": 0})
        self.assertEqual(200, first.status_code)
        version = first.json()["version_id"]
        self.assertTrue(version)
        latest = "<h2>Данные</h2><p>Первый абзац</p><p>Второй абзац</p><p>Третий абзац</p><table><tr><th>Показатель</th><th>Значение</th></tr><tr><td>Вес</td><td>12</td></tr></table>"
        saved = owner.post(note_save, json={**params, "html": latest, "revision": 1})
        self.assertEqual(200, saved.status_code)
        self.assertEqual(2, saved.json()["revision"])
        stale = owner.post(note_save, json={**params, "html": "<p>Older draft</p>", "revision": 1})
        self.assertEqual(409, stale.status_code)
        url = self.prefix + "/api/node/export_docx"
        exported = owner.get(url, params=params)
        self.assertEqual(200, exported.status_code)
        doc = Document(io.BytesIO(exported.content))
        self.assertTrue({"Первый абзац", "Второй абзац", "Третий абзац"} <= {p.text for p in doc.paragraphs})
        self.assertEqual([["Показатель", "Значение"], ["Вес", "12"]],
                         [[cell.text for cell in row.cells] for row in doc.tables[0].rows])
        historical = Document(io.BytesIO(owner.get(url, params={**params, "version_id": version}).content))
        self.assertIn("Archived draft", [x.text for x in historical.paragraphs])
        self.assertIn("Второй абзац", owner.get(route).text)
        self.assertEqual(2, len(owner.get(self.prefix + "/api/node/version/list", params=params).json()["versions"]))
        pdf = owner.get(self.prefix + "/api/node/export_pdf", params=params)
        self.assertEqual(200, pdf.status_code)
        self.assertTrue(pdf.content.startswith(b"%PDF-"))
        self.assertNotIn(b"/BaseFont /Symbol", pdf.content)
        self.assertIn(b"/FontFile2", pdf.content)
        pptx = owner.get(self.prefix + "/api/node/export_pptx", params=params)
        self.assertEqual(200, pptx.status_code)
        tables = [shape.table for slide in Presentation(io.BytesIO(pptx.content)).slides
                  for shape in slide.shapes if shape.has_table]
        self.assertEqual("12", tables[0].cell(1, 1).text)

    def test_protected_experiment_and_public_stage(self):
        owner = self.request("owner")
        member = self.request("member")
        guest = self.request()
        exp, node, route = self.experiment(owner)
        self.assertEqual(404, member.get(route).status_code)
        self.assertEqual(404, member.get(self.prefix + "/api/node/version/list",
                                       params={"exp_id": exp, "node_id": node}).status_code)
        self.assertEqual(302, owner.post(self.prefix + f"/experiment/owner/{exp}/toggle_public",
                                          follow_redirects=False).status_code)
        listing = guest.get(self.prefix + f"/share/{exp}")
        self.assertEqual(200, listing.status_code)
        href = f"{self.prefix}/share/{exp}/node/{node}"
        self.assertIn(f'href="{href}"', listing.text)
        self.assertEqual(200, guest.get(href).status_code)

    def test_image_access_and_attachment_quota(self):
        owner = self.request("owner")
        member = self.request("member")
        exp, node, route = self.experiment(owner)
        with self.m.SessionLocal() as db:
            before = self.m._user_storage_bytes("owner", db)
        out = owner.post(self.prefix + f"/api/experiment/{exp}/files/upload", files={
            "file": ("test.txt", b"example file", "text/plain")})
        self.assertEqual(200, out.status_code)
        file_id = out.json()["file_id"]
        with self.m.SessionLocal() as db:
            self.assertEqual(before + 12, self.m._user_storage_bytes("owner", db))
        self.assertEqual(404, member.get(self.prefix + f"/api/experiment/{exp}/files/{file_id}/download").status_code)
        fixture = io.BytesIO()
        Image.new("RGB", (2, 2)).save(fixture, format="PNG")
        image = owner.post(self.prefix + "/api/upload", data={"exp_id": exp}, files={
            "file": ("photo.png", fixture.getvalue(), "image/png")})
        self.assertEqual(200, image.status_code)
        url = image.json()["url"]
        self.assertEqual(200, owner.get(url).status_code)
        self.assertEqual(404, member.get(url).status_code)
        self.assertEqual(404, self.request().get(url).status_code)
        with self.m.SessionLocal() as db:
            self.assertEqual(before + 12 + len(fixture.getvalue()), self.m._user_storage_bytes("owner", db))
        guest = self.request()
        publish = self.prefix + f"/experiment/owner/{exp}/toggle_public"
        self.assertEqual(302, owner.post(publish, follow_redirects=False).status_code)
        self.assertEqual(404, guest.get(url).status_code)
        self.assertEqual(200, owner.post(self.prefix + "/api/node/save", json={
            "exp_id": exp, "node_id": node, "html": f'<img src="{url}" alt="Fixture">', "revision": 0,
        }).status_code)
        self.assertEqual(fixture.getvalue(), guest.get(url).content)
        self.assertEqual(302, owner.post(publish, follow_redirects=False).status_code)
        self.assertEqual(404, guest.get(url).status_code)
        admin = self.admin()
        remove = admin.post(self.prefix + self.m.LAB_ADMIN_PATH + "/delete_experiment", json={"exp_id": exp})
        self.assertEqual(200, remove.status_code)
        self.assertFalse(self.m._media_path(url.rsplit("/", 1)[-1]))
        with self.m.SessionLocal() as db:
            self.assertFalse(db.query(self.m.ExperimentFileDB).filter_by(exp_id=exp).first())
            self.assertFalse(db.query(self.m.NodeNotesDB).filter_by(exp_id=exp).first())

    def test_membership_and_task_dates(self):
        owner, member, observer = (self.request(name) for name in ("owner", "member", "observer"))
        project = owner.post(self.prefix + "/api/projects", json={"name": "Audit project"}).json()["id"]
        url = self.prefix + "/api/projects/" + project
        self.assertEqual(404, observer.get(url).status_code)
        invitation = owner.post(url + "/invite", json={}).json()["code"]
        self.assertEqual(200, member.post(self.prefix + f"/api/invite/{invitation}/accept", json={}).status_code)
        self.assertEqual("owner", owner.get(url).json()["invites"][0]["created_by"])
        self.assertEqual([], member.get(url).json()["invites"])
        index = member.get(self.prefix + "/api/projects", params={"user": "member"}).json()["items"]
        self.assertIn(project, {row["id"] for row in index})
        invalid = member.post(url + "/tasks", json={"title": "Task", "start_date": "2027-03-10", "end_date": "2027-03-07"})
        self.assertEqual(422, invalid.status_code)
        valid = member.post(url + "/tasks", json={"title": "Task", "start_date": "2027-03-07", "end_date": "2027-03-10"})
        self.assertEqual(200, valid.status_code)
        self.assertEqual(404, member.delete(url).status_code)

    def test_booking_overlap_and_identity(self):
        owner = self.request("owner")
        base = self.prefix + "/api/equipment"
        item = owner.post(base + "/add", json={"name": "Fixture device"}).json()["id"]
        ids = []
        for begin, end in (("2027-04-01", "2027-04-04"), ("2027-04-03", "2027-04-05")):
            booking = owner.post(base + "/book", json={"equipment_id": item, "start_date": begin, "end_date": end})
            self.assertEqual(200, booking.status_code)
            ids.append(booking.json()["id"])
        self.assertEqual(200, owner.post(base + "/booking/approve", json={"id": ids[0]}).status_code)
        self.assertEqual(409, owner.post(base + "/booking/approve", json={"id": ids[1]}).status_code)

    def test_logout_and_ban_revoke_sessions(self):
        owner = self.request("owner")
        old = owner.cookies[self.m.SESSION_COOKIE]
        owner.get(self.prefix + "/logout", follow_redirects=False)
        with self.m.SessionLocal() as db:
            self.assertIsNone(db.get(self.m.LoginSessionDB, self.m._session_digest(old)))
        member = self.request("member")
        admin = self.admin()
        base = self.prefix + self.m.LAB_ADMIN_PATH
        r = admin.post(base + "/api/user/edit", json={"username": "member", "is_banned": "true"})
        self.assertEqual(200, r.status_code)
        self.assertEqual(401, member.get(self.prefix + "/api/projects", params={"user": "member"}).status_code)
        self.assertEqual(200, admin.post(base + "/api/user/edit", json={"username": "member", "is_banned": "false"}).status_code)
        self.assertEqual(401, member.get(self.prefix + "/api/projects", params={"user": "member"}).status_code)
        anonymous = self.request()
        self.assertEqual(403, anonymous.get(base + "/api/user_dashboard").status_code)

    def test_chat_follows_friendship(self):
        owner, member = (self.request(name) for name in ("owner", "member"))
        api = self.prefix + "/api/friends"
        pair = {"user_id": "owner", "friend_id": "member"}
        self.assertEqual(200, owner.post(api + "/request", json=pair).status_code)
        self.assertEqual(200, member.post(api + "/accept", json={"friend_id": "owner"}).status_code)
        chat = owner.post(api + "/chats/create", json={"other": "member"})
        self.assertEqual(200, chat.status_code)
        room = api + "/chats/" + chat.json()["chat_id"]
        self.assertEqual(200, owner.post(room + "/send", json={"content": "Hello"}).status_code)
        self.assertEqual(200, owner.post(api + "/remove", json=pair).status_code)
        self.assertEqual(403, owner.post(room + "/send", json={"content": "Follow-up"}).status_code)
        self.assertEqual("Hello", member.get(room + "/messages").json()["messages"][0]["content"])

    def test_kernel_fails_closed_when_sandbox_cannot_start(self):
        with mock.patch.object(self.m, "_kernel_sandbox_command", return_value=None):
            output, failed = self.m._run_kernel_subprocess("print(42)", self.temp.name, {})
        self.assertTrue(failed)
        self.assertIn("недоступна", output)
        refusal = subprocess.CompletedProcess(["bwrap"], 1, stdout=b"", stderr=b"Operation not permitted")
        with mock.patch.object(self.m, "_kernel_sandbox_command", return_value=["bwrap"]),                 mock.patch.object(self.m.subprocess, "run", return_value=refusal),                 self.assertLogs("app", level="WARNING"):
            output, failed = self.m._run_kernel_subprocess("print(42)", self.temp.name, {})
        self.assertTrue(failed)
        self.assertIn("остановлено", output)

    def test_user_search_respects_email_privacy(self):
        observer = self.request("observer")
        result = observer.get(self.prefix + "/api/users/search", params={"q": "owner"})
        self.assertEqual(200, result.status_code)
        self.assertEqual([{"username": "owner"}], result.json()["users"])


if __name__ == "__main__":
    unittest.main()
