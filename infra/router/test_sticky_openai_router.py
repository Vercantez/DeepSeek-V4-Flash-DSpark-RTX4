import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sticky_openai_router as router


JSON_HEADERS = {"content-type": "application/json"}


class RequestDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.default_params = router.DEFAULT_REQUEST_PARAMS
        self.output_cap = router.MAX_REQUEST_OUTPUT_TOKENS
        router.DEFAULT_REQUEST_PARAMS = {
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_tokens": 1024,
        }
        router.MAX_REQUEST_OUTPUT_TOKENS = 512

    def tearDown(self):
        router.DEFAULT_REQUEST_PARAMS = self.default_params
        router.MAX_REQUEST_OUTPUT_TOKENS = self.output_cap
        router.STATE.replace([], [])

    def apply(self, path, payload, headers=None):
        encoded = json.dumps(payload).encode()
        result = router.apply_request_defaults(
            path,
            headers or JSON_HEADERS,
            encoded,
        )
        return json.loads(result)

    def test_chat_completions_gets_vllm_priority_and_chat_fields(self):
        result = self.apply(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hello"}]},
            {
                **JSON_HEADERS,
                router.VLLM_PRIORITY_HEADER: "7",
            },
        )

        self.assertEqual(result["priority"], 7)
        self.assertEqual(result["max_tokens"], 512)
        self.assertEqual(result["repetition_penalty"], 1.05)

    def test_responses_maps_output_budget_and_creates_affinity_id(self):
        result = self.apply(
            "/v1/responses",
            {"input": "hello", "max_output_tokens": 900},
        )

        self.assertEqual(result["max_output_tokens"], 512)
        self.assertEqual(result["priority"], router.DEFAULT_VLLM_PRIORITY)
        self.assertNotIn("max_tokens", result)
        self.assertNotIn("repetition_penalty", result)
        self.assertTrue(result["request_id"].startswith("resp_"))

        initial_key = router.sticky_key(JSON_HEADERS, json.dumps(result).encode())
        followup_key = router.sticky_key(
            JSON_HEADERS,
            json.dumps({"previous_response_id": result["request_id"]}).encode(),
        )
        self.assertEqual(initial_key, followup_key)

    def test_stateless_responses_do_not_get_a_request_id(self):
        result = self.apply(
            "/v1/responses",
            {"input": "hello", "store": False},
        )

        self.assertNotIn("request_id", result)

    def test_stored_response_ids_keep_the_chain_on_one_backend(self):
        backends = ["http://10.0.0.1:8000", "http://10.0.0.2:8000"]
        router.STATE.replace(backends, backends)

        initial = self.apply("/v1/responses", {"input": "hello"})
        initial_body = json.dumps(initial).encode()
        backend = router.STATE.choose(router.sticky_key(JSON_HEADERS, initial_body))
        self.assertIsNotNone(backend)
        initial_body = router.align_response_id_affinity(
            "/v1/responses", JSON_HEADERS, initial_body, backend
        )
        initial_id = json.loads(initial_body)["request_id"]

        followup = self.apply(
            "/v1/responses",
            {"input": "next", "previous_response_id": initial_id},
        )
        followup_body = json.dumps(followup).encode()
        followup_backend = router.STATE.choose(
            router.sticky_key(JSON_HEADERS, followup_body)
        )
        self.assertEqual(followup_backend, backend)
        followup_body = router.align_response_id_affinity(
            "/v1/responses", JSON_HEADERS, followup_body, followup_backend
        )
        followup_id = json.loads(followup_body)["request_id"]
        self.assertTrue(router.STATE.routes_to(f"response:{followup_id}", backend))

    def test_anthropic_messages_only_gets_supported_fields(self):
        result = self.apply(
            "/v1/messages",
            {
                "model": "deepseek-v4-flash-dspark",
                "max_tokens": 900,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(result["max_tokens"], 512)
        self.assertNotIn("priority", result)
        self.assertNotIn("repetition_penalty", result)


class StickyKeyTests(unittest.TestCase):
    def key(self, payload):
        return router.sticky_key(JSON_HEADERS, json.dumps(payload).encode())

    def test_prefers_responses_prompt_cache_key(self):
        self.assertEqual(
            self.key({"prompt_cache_key": "conversation-123", "input": "hello"}),
            "prompt-cache:conversation-123",
        )

    def test_uses_anthropic_metadata_user_id(self):
        self.assertEqual(
            self.key({"metadata": {"user_id": "user-123"}}),
            "user:user-123",
        )

    def test_skips_shared_system_prompt_for_first_user_message(self):
        first = self.key(
            {
                "messages": [
                    {"role": "system", "content": "shared system prompt"},
                    {"role": "user", "content": "unique task"},
                ]
            }
        )
        second = self.key(
            {
                "messages": [
                    {"role": "system", "content": "shared system prompt"},
                    {"role": "user", "content": "another task"},
                ]
            }
        )

        self.assertNotEqual(first, second)

    def test_reads_responses_content_parts(self):
        plain = self.key({"input": "same task"})
        structured = self.key(
            {
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "same task"}],
                    }
                ]
            }
        )

        self.assertEqual(plain, structured)


if __name__ == "__main__":
    unittest.main()
