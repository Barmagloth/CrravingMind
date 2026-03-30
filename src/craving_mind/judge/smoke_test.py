from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from craving_mind.agent.sandbox import Sandbox


class SmokeTest:
    """Gate check: does compress.py run without crashing on 10 sample texts?"""

    SAMPLE_TEXTS = [
        "The quick brown fox jumps over the lazy dog. This is a simple test sentence that should be easy to compress.",
        "In 2024, the global GDP reached approximately $105 trillion. The United States contributed $28.78 trillion, while China contributed $18.53 trillion.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Article 7.2: The licensee shall maintain all records for a period of not less than five (5) years from the date of termination.",
        "The patient presented with acute respiratory distress. BP: 140/90, HR: 110, SpO2: 88%. Administered 2L O2 via nasal cannula.",
        "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name HAVING COUNT(o.id) > 5;",
        "According to the meeting minutes from March 15th, the committee agreed to allocate $2.5M to the infrastructure project, with $1.2M designated for Phase 1.",
        "The experiment yielded a p-value of 0.003 (n=450), suggesting a statistically significant correlation between variables X and Y (r=0.72, CI: 0.65-0.79).",
        "First, preheat the oven to 350°F. Then, mix 2 cups flour, 1 cup sugar, and 3 eggs. Bake for 25-30 minutes until golden brown.",
        "ERROR: NullPointerException at com.app.service.UserService.getProfile(UserService.java:142)\n    at com.app.controller.ApiController.handleRequest(ApiController.java:89)",
    ]

    def __init__(self, sandbox: "Sandbox"):
        self.sandbox = sandbox

    def run(self, compress_code: str) -> tuple[bool, list[str]]:
        """Run smoke test. Returns (all_passed, list_of_errors)."""
        errors = []

        for i, text in enumerate(self.SAMPLE_TEXTS):
            result = self.sandbox.run_compress(compress_code, text, 0.5)
            if not result.success:
                errors.append(f"Sample {i}: {result.error}")
            elif result.return_value is not None and len(result.return_value) == 0:
                errors.append(f"Sample {i}: compress() returned empty string")

        return len(errors) == 0, errors
