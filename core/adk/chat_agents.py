"""ADK Chat Pipeline: Router → Specialist → Judge → Beautifier."""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from core.tools.sector_tools import get_sector_rankings, get_sector_detail
from core.tools.company_tools import (
    get_best_companies,
    get_company_detail,
    compare_companies,
    search_similar_companies,
)
from core.tools.portfolio_tools import (
    optimize_portfolio,
    analyze_portfolio,
    get_pareto_frontier,
)
from core.tools.knowledge_tools import search_knowledge_base

# ════════════════════════════════════════════════════════════
# SPECIALIST AGENTS
# ════════════════════════════════════════════════════════════

sector_agent = LlmAgent(
    name="SectorAnalyst",
    model="gemini-2.5-flash",
    description="Analyzes and ranks market sectors by Sharpe ratio, ESG score, and risk metrics.",
    instruction="""You are a sector analysis specialist for an ESG investment platform.
    Use your tools to query sector rankings and provide data-backed analysis.
    Always include: Sharpe ratio, average ESG score, volatility, company count.
    Present data in structured format for downstream agents.""",
    tools=[
        FunctionTool(get_sector_rankings),
        FunctionTool(get_sector_detail),
    ],
    output_key="specialist_result",
)

company_agent = LlmAgent(
    name="CompanyAnalyst",
    model="gemini-2.5-flash",
    description="Analyzes individual companies on financial metrics, ESG scores, risk, and finds similar companies.",
    instruction="""You are a company analysis specialist. Use your tools to retrieve
    company data, compare companies, and find similar companies via vector search.
    Always present complete metrics: Sharpe, ESG breakdown (E/S/G), volatility,
    max drawdown, sector rank. Use tables for comparisons.""",
    tools=[
        FunctionTool(get_best_companies),
        FunctionTool(get_company_detail),
        FunctionTool(compare_companies),
        FunctionTool(search_similar_companies),
    ],
    output_key="specialist_result",
)

portfolio_agent = LlmAgent(
    name="PortfolioOptimizer",
    model="gemini-2.5-flash",
    description="Builds optimal investment portfolios using Deep Reinforcement Learning and analyzes proposed portfolios.",
    instruction="""You are a portfolio optimization specialist powered by a Deep
    Reinforcement Learning engine. Use your tools to generate optimal portfolio
    allocations and analyze proposed portfolios. Always explain:
    - Risk/return/ESG trade-offs
    - Diversification across sectors
    - Individual position weights and why
    - Portfolio-level metrics (Sharpe, volatility, ESG score)""",
    tools=[
        FunctionTool(optimize_portfolio),
        FunctionTool(analyze_portfolio),
        FunctionTool(get_pareto_frontier),
    ],
    output_key="specialist_result",
)

education_agent = LlmAgent(
    name="InvestmentEducator",
    model="gemini-2.5-flash",
    description="Explains investment concepts, ESG methodology, Sharpe ratio, diversification, and how the system works.",
    instruction="""You are an investment education specialist. Search the knowledge
    base for accurate definitions and explanations. Use simple language with
    real examples. Define technical terms when first used. Target audience
    is investors who may not be familiar with quantitative finance.""",
    tools=[
        FunctionTool(search_knowledge_base),
    ],
    output_key="specialist_result",
)

# ════════════════════════════════════════════════════════════
# ROUTER AGENT
# ════════════════════════════════════════════════════════════

router_agent = LlmAgent(
    name="InvestmentAdvisorRouter",
    model="gemini-2.5-flash",
    description="Main coordinator that routes investor queries to the right specialist agent.",
    instruction="""You are the coordinator of an ESG investment advisory system.
    Analyze the user's query and delegate to the right specialist:

    - Sector questions ("best sectors", "which industry") → SectorAnalyst
    - Company questions, comparisons ("tell me about AAPL", "compare X and Y") → CompanyAnalyst
    - Portfolio building, optimization ("build me a portfolio", "invest $50K") → PortfolioOptimizer
    - Concept questions ("what is Sharpe ratio", "how does ESG work") → InvestmentEducator

    For complex queries like "Build a tech-focused portfolio with high ESG":
    delegate to PortfolioOptimizer (it can handle sector preferences).

    Always delegate — never answer directly. Let the specialists use their tools.""",
    sub_agents=[sector_agent, company_agent, portfolio_agent, education_agent],
)

# ════════════════════════════════════════════════════════════
# JUDGE AGENT
# ════════════════════════════════════════════════════════════

judge_agent = LlmAgent(
    name="QualityJudge",
    model="gemini-2.5-pro",
    description="Validates accuracy, completeness, and quality of specialist responses.",
    instruction="""You are a quality control judge for investment advice responses.
    Read the specialist result from state key 'specialist_result'.

    CHECK:
    1. ACCURACY: Are numbers reasonable? Sharpe ratios typically 0-3, ESG 0-100,
       volatility 5-40%, max drawdown -5% to -60%.
    2. COMPLETENESS: Does it answer what the user actually asked?
    3. DATA-BACKED: Is everything based on tool-retrieved data, not fabricated?
    4. RISK DISCLOSURE: Are trade-offs, limitations, and risks mentioned?
    5. BALANCE: Does it present both strengths and weaknesses?

    If ALL checks pass: Output the specialist result as-is, prefixed with "APPROVED: "
    If ISSUES found: Output "NEEDS_REVISION: " followed by specific corrections needed,
    then include the original response with corrections applied.

    Do NOT reformat — only validate and correct factual issues.""",
    output_key="judged_result",
)

# ════════════════════════════════════════════════════════════
# RESPONSE BEAUTIFIER
# ════════════════════════════════════════════════════════════

response_agent = LlmAgent(
    name="ResponseBeautifier",
    model="gemini-2.5-flash",
    description="Formats the validated response into a beautiful, investor-friendly format.",
    instruction="""You are a response formatting specialist for an investor-facing chatbot.
    Read the judged result from state key 'judged_result'.
    Strip any "APPROVED:" or "NEEDS_REVISION:" prefixes.

    FORMAT RULES:
    1. Use markdown: ## headers, **bold** metrics, tables for comparisons
    2. Format numbers: Sharpe: **1.85**, ESG: **74/100**, Return: **12.3%**
    3. Use tables for multi-company or multi-sector data
    4. Use bullet points for recommendations
    5. Keep language professional but accessible
    6. Keep it concise — investors want clarity, not essays
    7. If portfolio advice was given, end with:
       ---
       *Disclaimer: This is AI-generated analysis for informational purposes only.
       Not financial advice. Past performance does not guarantee future results.
       Consult a qualified financial advisor before making investment decisions.*

    Do NOT change any data, numbers, or recommendations — only improve formatting.""",
)

# ════════════════════════════════════════════════════════════
# ROOT PIPELINE
# ════════════════════════════════════════════════════════════

root_agent = SequentialAgent(
    name="ESGAdvisorPipeline",
    description="Full multi-agent pipeline: route → specialize → judge → beautify",
    sub_agents=[router_agent, judge_agent, response_agent],
)
