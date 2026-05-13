import pytest
import yaml

@pytest.fixture
def env_config():
    with open("env_config.yml") as f:
        return yaml.safe_load(f)
    
class TestEnvConfig:
    def test_loads_without_error(self, env_config):
        assert env_config is not None
        
    def test_has_project_name(self, env_config):
        assert "project_name" in env_config
    
    def test_has_local_config(self, env_config):
        assert "snowflake_local" in env_config
    
    def test_local_has_required_fields(self, env_config):
        required = {"account","user","role","warehouse","database","schema"}
        assert required.issubset(set(env_config["snowflake_local"].keys()))
        
    def test_local_schema_references_project_name(self, env_config):
        assert env_config["snowflake_local"]["schema"] == env_config["project_name"]