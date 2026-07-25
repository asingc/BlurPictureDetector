"""LLM-assisted processing (currently: automatic burst culling).

Sub-modules
-----------
prompts:
    Provider-agnostic prompt text shared by every backend, so switching
    providers (OpenAI today, others later) never changes what is asked.
culling_provider:
    :class:`CullingProvider` — the ABC every backend implements — plus the
    OpenAI-backed :class:`OpenAIProvider`.
"""
