"""Single source of truth for the platform version.
Bump on every release; agents and coordinator report this everywhere
(/status, heartbeats, doctor bundles) so support always knows what's deployed.
"""
VERSION = "0.1.0"
