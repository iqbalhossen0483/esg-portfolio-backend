"""ADK callbacks for safety guardrails and tool validation."""


def input_safety_callback(callback_context, llm_request):
    """Block non-investment queries and inappropriate content.
    Returns a canned response if blocked, None to proceed normally.
    """
    try:
        user_msg = ""
        if llm_request.contents:
            last = llm_request.contents[-1]
            if last.parts:
                user_msg = last.parts[0].text.lower() if last.parts[0].text else ""
    except (AttributeError, IndexError):
        return None

    if not user_msg:
        return None

    blocked_phrases = [
        "crypto prediction", "guaranteed returns", "insider trading",
        "pump and dump", "get rich quick", "hack", "exploit",
    ]

    for phrase in blocked_phrases:
        if phrase in user_msg:
            from google.genai import types
            return types.GenerateContentResponse(
                candidates=[types.Candidate(
                    content=types.Content(parts=[types.Part(
                        text="I can only help with S&P 500 ESG investment analysis. "
                             "I cannot provide predictions, guaranteed returns, or advice "
                             "on restricted topics."
                    )])
                )]
            )

    return None


def tool_validation_callback(callback_context, tool_name, tool_args):
    """Validate tool arguments before execution.
    Returns a dict to skip tool execution, None to proceed.
    """
    if tool_name == "optimize_portfolio":
        amount = tool_args.get("investment_amount")
        if amount is not None and amount < 0:
            return {"error": "Investment amount must be positive"}
        max_stocks = tool_args.get("max_stocks", 15)
        if max_stocks > 30:
            tool_args["max_stocks"] = 30

    if tool_name in ("get_sector_rankings", "get_best_companies"):
        top_n = tool_args.get("top_n", 10)
        if top_n > 50:
            tool_args["top_n"] = 50

    return None
