from config import EnvironmentSettings, run_preflight


def test_preflight_reports_only_selected_integration_requirements() -> None:
    settings = EnvironmentSettings(
        tavily_api_key="",
        llm_api_key="",
        neo4j_uri="",
        neo4j_password="",
        label_studio_url="",
        label_studio_api_key="",
        label_studio_project_id=None,
    )

    result = run_preflight(["tavily"], settings=settings)

    assert not result.ok
    assert result.missing_variables == ["TAVILY_API_KEY"]
    assert "NEO4J_URI" not in result.missing_variables


def test_preflight_accepts_ready_selected_integration() -> None:
    settings = EnvironmentSettings(tavily_api_key="test-key")

    result = run_preflight(["tavily"], settings=settings)

    assert result.ok
    assert result.missing_variables == []
