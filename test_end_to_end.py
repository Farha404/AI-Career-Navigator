"""Offline integration tests for the semantic matcher used by app.py.

These test the production profile-to-ranking path using deterministic NLP
fixtures. Live API access and serialized model compatibility are separate
integration concerns; no network or model artifact is required here.
"""
import unittest
from types import SimpleNamespace

from sklearn.feature_extraction.text import TfidfVectorizer

from src.semantic_matcher import (
    build_extended_candidate_profile,
    semantic_match_cv_to_jobs,
)


class UnusedClassifier:
    def predict(self, *args, **kwargs):
        raise AssertionError("Semantic ranking must not claim classifier output")

    predict_proba = predict


class ProductionMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = [
            "software engineer python sql docker rest api software development",
            "sales representative outbound crm sales targets",
        ]
        cls.bundle = SimpleNamespace(
            SKILL_VOCAB=["Python", "SQL", "Docker", "REST APIs"],
            SKILL_ALIASES={},
            tfidf_text=TfidfVectorizer().fit(corpus),
            tfidf_title=TfidfVectorizer().fit(corpus),
            canonical_role=lambda title: title,
            model=UnusedClassifier(),
        )
        cls.candidate = build_extended_candidate_profile(
            """Jane Smith
Software Engineer
Skills: Python, SQL, REST APIs
EDUCATION
BSc, Example University | 2018 - 2022
WORK EXPERIENCE
Software Engineer at Acme | Jan 2023 - Dec 2024
Developed Python services and SQL queries.
""",
            cls.bundle,
        )
        cls.jobs = [
            {
                "slug": "software", "title": "Software Engineer",
                "description": "Requires 3 years of experience. Python SQL Docker REST APIs.",
                "tags": ["Python", "SQL", "Docker"], "company": "Acme",
            },
            {
                "slug": "sales", "title": "Sales Representative",
                "description": "Outbound sales and CRM; meet sales targets.",
                "tags": [], "company": "Other",
            },
        ]
        cls.results = semantic_match_cv_to_jobs(cls.candidate, cls.jobs, cls.bundle)

    def test_education_does_not_inflate_work_experience(self):
        self.assertEqual(self.candidate["years_experience"], 2)

    def test_production_path_returns_every_fixture_job(self):
        self.assertEqual({r["job_id"] for r in self.results}, {"software", "sales"})

    def test_score_schema_and_threshold(self):
        for result in self.results:
            self.assertGreaterEqual(result["match_score"], 0)
            self.assertLessEqual(result["match_score"], 1)
            self.assertAlmostEqual(result["match_score"] * 100, result["match_score_pct"])
            self.assertEqual(result["matched"], result["match_score_pct"] >= 50)
            for field in ("match_probability", "match_prob_pct", "confidence", "confidence_pct"):
                self.assertNotIn(field, result)

    def test_ranking_and_domain_guard(self):
        scores = [r["match_score"] for r in self.results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        by_id = {r["job_id"]: r for r in self.results}
        self.assertGreater(by_id["software"]["match_score"], by_id["sales"]["match_score"])
        self.assertLessEqual(by_id["sales"]["match_score_pct"], 18)

    def test_skill_gap_and_job_requirement(self):
        result = next(r for r in self.results if r["job_id"] == "software")
        self.assertIn("Python", result["shared_skills"])
        self.assertIn("Docker", result["missing_skills"])
        self.assertEqual(result["job_required_years"], 3)

    def test_explanations_and_empty_jobs(self):
        self.assertTrue(all(r["match_reason"] for r in self.results))
        self.assertEqual(semantic_match_cv_to_jobs(self.candidate, [], self.bundle), [])


if __name__ == "__main__":
    unittest.main()
