import argparse
import importlib
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

    
def resolve_function(func_path: str):
    module_path, func_name = func_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def resolve_reference(ref: str, context: dict):
    parts = ref.split(".")
    result = context
    for part in parts:
        if isinstance(result, dict) and part in result:
            result = result[part]
        else:
            raise KeyError(f"Cannot resolve reference '${ref}' - '{part}' not found")
    return result


def resolve_params(params: dict, context: dict) -> dict:
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("$"):
            resolved[key] = resolve_reference(value[1:], context)
        elif isinstance(value, list):
            resolved[key] = [
                resolve_reference(item[1:], context) if isinstance(item, str) and item.startswith("$") else item
                for item in value
            ]
        else:
            resolved[key] = value
    return resolved


def run_step(step_config: dict, context: dict) -> dict:
    func = resolve_function(step_config["function"])
    params = resolve_params(step_config.get("params",{}), context)
    return func(**params)
   


def run_section(section_name: str, section: dict, context: dict) -> dict:    
    results = {}
    context[section_name] = results
    for step_name, steps in section.items():
        if step_name == "scope":
            continue
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict) and "function" in step:
                    results[step_name] = run_step(step, context)
                    print(f"    ✅ {step_name}")


def run(config_path: str, env: str):
    config = load_config(config_path)
    all_env_configs = load_config("env_config.yml")
    # chosen_env_config = all_env_configs.get(env, {})
    if env not in all_env_configs:
        valid_keys = [k for k in all_env_configs if k != "project_name"]
        raise KeyError(f"Unknown env '{env}'. Valid keys: {valid_keys}")
    chosen_env_config = all_env_configs[env]
    context = {
        "env_config": chosen_env_config,
        "model_version": config.get("model_version")
    }
    
    reserved_keys = {"env_config", "model_version"}
    sections = [key for key in config.keys() if key not in reserved_keys]
    print("===================================================")
    print(f"------------- Environment Details ----------------")
    print("===================================================")
    print(f"Project:        {all_env_configs.get('project_name', 'N/A')}")
    print(f"Model Version:  {config.get('model_version', 'N/A')}")
    print(f"Environment:    {env}")
    print()
    
    for section_name in sections:
        section = config[section_name]
        if not isinstance(section, dict):
            continue
        print("===================================================")
        print(f"----------- STEP: {section_name} -------------")
        print("===================================================")
        run_section(section_name, section, context)
        print()
    
    print("...all done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Pipeline Runner")
    parser.add_argument("--model_version", required=True, help="model version YAML filename (e.g. v1.yaml)")
    parser.add_argument("--env", required=True, help="Environment key in env_config.yml (e.g. snowflake_local)")
    args = parser.parse_args()
    run(config_path=args.model_version, env=args.env)