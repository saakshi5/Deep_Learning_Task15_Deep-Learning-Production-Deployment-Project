
import unittest
import app as api_module


class APITests(unittest.TestCase):

    def setUp(self):
        self.client = api_module.app.test_client()

    def test_home_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertIn(response.status_code, [200, 503])

    def test_model_info_endpoint(self):
        response = self.client.get("/model-info")
        self.assertIn(response.status_code, [200, 503])

    def test_predict_rejects_non_json(self):
        response = self.client.post(
            "/predict",
            data="not json",
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 415)

    def test_predict_rejects_empty_text(self):
        response = self.client.post(
            "/predict",
            json={"text": "   "},
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_rejects_missing_text(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)

    def test_predict_rejects_non_string(self):
        response = self.client.post(
            "/predict",
            json={"text": 123},
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_rejects_long_text(self):
        response = self.client.post(
            "/predict",
            json={"text": "a" * 2001},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
