from moe_congestion_routing.main import greet


def test_greet_default():
    assert greet() == "Hello from moe-congestion-routing!"


def test_greet_with_name():
    assert greet("world") == "Hello from world!"
