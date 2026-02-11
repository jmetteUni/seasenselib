from seasenselib.cli.commands.info_commands import ListCommand


def test_list_pipeline_profiles_includes_descriptions():
    cmd = ListCommand(None)
    profiles = cmd._list_pipeline_profiles()
    names = {item["name"]: item for item in profiles}

    assert "default" in names
    assert "minimal" in names
    assert "full" in names

    assert names["default"]["description"]
    assert names["minimal"]["description"]
    assert names["full"]["description"]
