from dataset_recsys.mathe_recommenders.seed_scoring import (
    compute_keyword_jaccard,
    score_document_seed_candidates,
)


def test_compute_keyword_jaccard_handles_empty_values_and_duplicates():
    assert compute_keyword_jaccard(None, None) == 0.0
    assert compute_keyword_jaccard(["algebra"], None) == 0.0
    assert compute_keyword_jaccard(
        ["algebra", "matrix", "matrix"],
        ["matrix", "calculus"],
    ) == 1 / 3


def test_score_document_seed_candidates_enriches_and_sorts_by_metadata_score():
    question_metadata = {
        "question_id": 42,
        "topic_id": 10,
        "subtopic_id": 20,
        "keywords": ["algebra", "matrix"],
    }
    seed_candidates = [
        {
            "material_id": 1,
            "topic_ids": [10],
            "subtopic_ids": [99],
            "keywords": ["geometry"],
        },
        {
            "material_id": 2,
            "topic_ids": [10],
            "subtopic_ids": [20],
            "keywords": ["algebra", "matrix"],
        },
        {
            "material_id": 3,
            "topic_ids": [99],
            "subtopic_ids": [99],
            "keywords": ["matrix", "calculus"],
        },
    ]

    scored = score_document_seed_candidates(question_metadata, seed_candidates)

    assert [candidate["material_id"] for candidate in scored] == [2, 1, 3]
    assert scored[0]["keyword_jaccard"] == 1.0
    assert scored[0]["same_subtopic"] == 1
    assert scored[0]["same_topic"] == 1
    assert scored[0]["metadata_score"] == 1.0
    assert scored[1]["metadata_score"] == 1 / 3
    assert scored[2]["metadata_score"] == 1 / 9


def test_score_document_seed_candidates_matches_plural_topic_and_subtopic_ids():
    question_metadata = {
        "question_id": 42,
        "topic_id": 2,
        "subtopic_id": 3,
        "keywords": ["Derivatives"],
    }
    seed_candidates = [
        {
            "material_id": 818,
            "topic_ids": [1, 2],
            "subtopic_ids": [1, 3],
            "keywords": ["Derivatives", "Partial Differentiation"],
        }
    ]

    scored = score_document_seed_candidates(question_metadata, seed_candidates)

    assert scored[0]["same_topic"] == 1
    assert scored[0]["same_subtopic"] == 1
    assert scored[0]["metadata_score"] == (0.5 + 1 + 1) / 3
