"""
Synthetic corpus generator for RAG-RL.
Generates deterministic document corpora without requiring an LLM.
Documents are designed so sentence-transformer embeddings will retrieve
the correct chunks for each query (topic vocabulary overlap).
"""

import json
import os
import random
from typing import Any

from task_configs import TaskConfig


# ---------------------------------------------------------------------------
# 20 richly defined topics for TechNova Corp
# ---------------------------------------------------------------------------

TOPIC_DEFINITIONS = [
    # ── Single-hop topics (0-14) ────────────────────────────────────────────
    {
        "id": 0,
        "name": "vacation_policy",
        "query": "How many vacation days do TechNova Corp employees receive annually?",
        "reference_answer": (
            "TechNova Corp employees receive 15 days of paid vacation annually, "
            "increasing to 20 days after 5 years of service."
        ),
        "golden": (
            "TechNova Corp vacation policy: All full-time employees receive 15 paid "
            "vacation days per year. Employees with more than 5 years of continuous "
            "service are entitled to 20 paid vacation days annually."
        ),
        "redundants": [
            (
                "TechNova Corp grants 15 days of annual paid time off to full-time staff. "
                "After five years of employment, the vacation allowance rises to 20 days per year."
            ),
            (
                "At TechNova Corp, full-time employees enjoy 15 vacation days each year. "
                "This PTO benefit grows to 20 days annually for those who reach the 5-year tenure milestone."
            ),
            (
                "The paid leave policy at TechNova Corp provides 15 annual vacation days. "
                "Long-serving employees (5+ years) are rewarded with an additional 5 vacation days yearly."
            ),
        ],
        "contradiction": (
            "TechNova Corp vacation policy: Full-time employees receive only 8 paid vacation "
            "days per year. There are no tenure-based increases to vacation entitlement."
        ),
        "noise": (
            "TechNova Corp reported record Q3 revenue of $2.4 billion, up 18% year-over-year, "
            "driven by its cloud services division and international expansion into Southeast Asia."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 1,
        "name": "sick_leave",
        "query": "What is TechNova Corp's paid sick leave policy?",
        "reference_answer": (
            "TechNova Corp provides 10 days of paid sick leave per year, which can be used "
            "for personal illness or to care for an immediate family member."
        ),
        "golden": (
            "TechNova Corp sick leave policy: Full-time employees are entitled to 10 paid "
            "sick days per year. Sick leave may be used for the employee's own illness or "
            "to care for an immediate family member such as a spouse, child, or parent."
        ),
        "redundants": [
            (
                "TechNova Corp provides 10 days of paid sick leave annually. Employees may "
                "use sick days both for their own health needs and to care for immediate family members."
            ),
            (
                "At TechNova Corp, the sick leave benefit is 10 paid days per year. This "
                "covers personal illness as well as caregiving for a spouse, child, or parent."
            ),
            (
                "TechNova's paid sick leave entitlement is 10 days each year. The policy "
                "covers both personal medical needs and family caregiving responsibilities."
            ),
        ],
        "contradiction": (
            "TechNova Corp sick leave policy: Employees receive 5 days of paid sick leave "
            "annually. Sick leave may only be used for the employee's own illness, not for "
            "family caregiving."
        ),
        "noise": (
            "TechNova Corp's engineering team adopted a new CI/CD pipeline using GitHub Actions "
            "and ArgoCD, reducing deployment time from 45 minutes to under 8 minutes."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 2,
        "name": "remote_work",
        "query": "How many days per week can TechNova Corp employees work remotely?",
        "reference_answer": (
            "TechNova Corp employees can work remotely up to 3 days per week, subject to "
            "manager approval and role requirements."
        ),
        "golden": (
            "TechNova Corp remote work policy: Eligible employees may work from home up to "
            "3 days per week. Remote work arrangements are subject to manager approval and "
            "depend on role requirements. Core collaboration hours (10am–3pm local time) "
            "must be observed regardless of work location."
        ),
        "redundants": [
            (
                "TechNova Corp allows employees to work remotely for up to 3 days each week. "
                "Manager approval is required, and employees must be available during core hours."
            ),
            (
                "Under TechNova's hybrid work policy, staff can choose to work from home on "
                "3 days per week. This requires managerial sign-off and adherence to core hours."
            ),
            (
                "TechNova Corp supports a hybrid working model where employees can work remotely "
                "3 days per week with their manager's approval."
            ),
        ],
        "contradiction": (
            "TechNova Corp remote work policy: Employees are required to be in the office "
            "5 days per week. Remote work is not permitted except in exceptional circumstances "
            "approved by the VP of their department."
        ),
        "noise": (
            "TechNova Corp's annual hackathon attracted over 500 participants who built 87 "
            "projects over 48 hours, with the winning team creating an AI-powered accessibility tool."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 3,
        "name": "health_benefits",
        "query": "What health insurance benefits does TechNova Corp provide?",
        "reference_answer": (
            "TechNova Corp provides comprehensive health coverage including medical, dental, "
            "and vision insurance with zero premium cost for the employee."
        ),
        "golden": (
            "TechNova Corp health benefits: The company provides comprehensive medical, dental, "
            "and vision insurance at no premium cost to the employee. Dependent coverage is "
            "available at a subsidized rate. The medical plan includes mental health services, "
            "prescription drug coverage, and preventive care at $0 copay."
        ),
        "redundants": [
            (
                "TechNova Corp offers full medical, dental, and vision insurance free of charge "
                "to employees. Family members can be added to the plan at a subsidized rate."
            ),
            (
                "Employees at TechNova Corp receive zero-premium health coverage including "
                "medical, dental, and vision plans. Mental health and prescription coverage "
                "are included in the medical plan."
            ),
            (
                "TechNova Corp's health insurance package covers medical, dental, and vision "
                "at no cost to the employee, with subsidized dependent coverage available."
            ),
        ],
        "contradiction": (
            "TechNova Corp health benefits: Employees contribute $350 per month toward their "
            "health insurance premium. Dental and vision are offered as optional add-ons at "
            "additional cost."
        ),
        "noise": (
            "TechNova Corp's data platform processes over 5 petabytes of data daily using "
            "Apache Kafka and Apache Spark running on a Kubernetes cluster across 3 availability zones."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 4,
        "name": "401k_policy",
        "query": "What is TechNova Corp's 401k matching policy?",
        "reference_answer": (
            "TechNova Corp matches 100% of employee 401k contributions up to 6% of salary, "
            "with immediate vesting."
        ),
        "golden": (
            "TechNova Corp 401k policy: The company offers a 401k retirement plan with a "
            "generous employer match. TechNova matches 100% of employee contributions up to "
            "6% of the employee's base salary. The employer match vests immediately — there "
            "is no waiting period."
        ),
        "redundants": [
            (
                "TechNova Corp's 401k plan includes a dollar-for-dollar match on contributions "
                "up to 6% of base salary. The match vests immediately with no cliff or schedule."
            ),
            (
                "Employees at TechNova Corp benefit from a 6% salary match on their 401k "
                "contributions, with immediate vesting from day one of employment."
            ),
            (
                "TechNova Corp retirement benefits include a 401k with 100% match on the first "
                "6% of salary contributed. Vesting is immediate for all matched funds."
            ),
        ],
        "contradiction": (
            "TechNova Corp 401k policy: The company matches 50% of employee contributions "
            "up to 4% of salary. The employer match vests over a 3-year schedule."
        ),
        "noise": (
            "TechNova Corp unveiled its new open-source observability framework, which has "
            "already received 12,000 GitHub stars and contributions from 340 external developers."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 5,
        "name": "parental_leave",
        "query": "How much paid parental leave does TechNova Corp offer?",
        "reference_answer": (
            "TechNova Corp offers 16 weeks of paid parental leave for the primary caregiver "
            "and 4 weeks for the secondary caregiver."
        ),
        "golden": (
            "TechNova Corp parental leave policy: Primary caregivers (including birth parents "
            "and adoptive parents) receive 16 weeks of fully paid parental leave. Secondary "
            "caregivers receive 4 weeks of fully paid leave. Leave can be taken any time within "
            "the first year after birth or adoption."
        ),
        "redundants": [
            (
                "TechNova Corp provides 16 weeks paid leave for primary caregivers and "
                "4 weeks for secondary caregivers following birth or adoption."
            ),
            (
                "Under TechNova's parental leave policy, the primary caregiver receives 16 "
                "fully paid weeks off, while the secondary caregiver receives 4 paid weeks."
            ),
            (
                "TechNova Corp parental benefits include 16 weeks of paid leave for primary "
                "parents and 4 weeks for secondary parents, usable within the child's first year."
            ),
        ],
        "contradiction": (
            "TechNova Corp parental leave policy: All new parents receive a standard 6 weeks "
            "of paid parental leave, regardless of caregiver role."
        ),
        "noise": (
            "TechNova Corp's machine learning platform supports training models with up to "
            "70 billion parameters using distributed training across 512 H100 GPUs."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 6,
        "name": "performance_review",
        "query": "When does TechNova Corp conduct performance reviews?",
        "reference_answer": (
            "TechNova Corp conducts annual performance reviews in Q4 (October–November), "
            "with salary adjustments taking effect in January."
        ),
        "golden": (
            "TechNova Corp performance review cycle: Annual performance reviews are conducted "
            "in Q4, specifically during October and November. Managers submit evaluations by "
            "November 30. Salary adjustments based on performance ratings take effect on "
            "January 1 of the following year."
        ),
        "redundants": [
            (
                "TechNova conducts its annual performance reviews in the fourth quarter, "
                "October through November, with pay increases effective January 1."
            ),
            (
                "Performance evaluations at TechNova Corp happen annually in Q4. Review "
                "submissions close November 30, and any salary changes go into effect in January."
            ),
            (
                "TechNova's yearly review cycle runs through Q4 (Oct–Nov), and compensation "
                "adjustments resulting from reviews are applied at the start of the new year."
            ),
        ],
        "contradiction": (
            "TechNova Corp performance review cycle: Reviews are conducted twice yearly, "
            "in Q2 (April) and Q4 (October). Salary adjustments take effect immediately "
            "following each review cycle."
        ),
        "noise": (
            "TechNova Corp signed a strategic partnership with three Fortune 500 companies "
            "to integrate its API platform, projected to add $180 million in annual recurring revenue."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 7,
        "name": "learning_budget",
        "query": "What is TechNova Corp's professional development budget for employees?",
        "reference_answer": (
            "TechNova Corp provides $2,000 per year per employee for professional development, "
            "covering courses, certifications, books, and conferences."
        ),
        "golden": (
            "TechNova Corp learning and development budget: Each employee receives $2,000 "
            "per calendar year to spend on professional development. Eligible expenses include "
            "online courses, certification exams, technical books, and conference attendance. "
            "Unused budget does not roll over to the next year."
        ),
        "redundants": [
            (
                "TechNova Corp allocates $2,000 annually per employee for learning and "
                "professional growth, covering courses, certs, books, and conferences."
            ),
            (
                "Each TechNova Corp employee has access to a $2,000/year professional "
                "development stipend for courses, certifications, books, and events."
            ),
            (
                "TechNova's learning budget gives employees $2,000 per year to invest in "
                "professional development including online learning, exams, and conferences."
            ),
        ],
        "contradiction": (
            "TechNova Corp professional development: Employees receive $500 per year for "
            "professional development. This budget can only be used for pre-approved courses."
        ),
        "noise": (
            "TechNova Corp's global workforce grew to 12,500 employees across 28 countries "
            "following its acquisition of DataStream Inc. in Q2."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 8,
        "name": "stock_options",
        "query": "How do TechNova Corp stock options vest?",
        "reference_answer": (
            "TechNova Corp stock options vest over 4 years with a 1-year cliff, meaning "
            "25% vests after year one and the remainder vests monthly over the following 3 years."
        ),
        "golden": (
            "TechNova Corp equity compensation: Stock options are granted with a standard "
            "4-year vesting schedule and a 1-year cliff. Employees who leave before one year "
            "receive no vested options. After the cliff, 25% of the grant vests, and the "
            "remaining 75% vests monthly over the next 36 months."
        ),
        "redundants": [
            (
                "TechNova stock options vest on a 4-year schedule with a 1-year cliff. "
                "25% of options vest on the 1-year anniversary, then monthly for 3 years."
            ),
            (
                "TechNova Corp equity vests over 4 years (1-year cliff). After the cliff, "
                "a quarter of the grant vests immediately and the rest vests monthly."
            ),
            (
                "TechNova's stock option vesting: 4-year total with a 1-year cliff. "
                "One quarter vests after year one; the remaining three quarters vest monthly."
            ),
        ],
        "contradiction": (
            "TechNova Corp equity: Stock options vest over 2 years with no cliff. Options "
            "vest in equal monthly installments starting from the grant date."
        ),
        "noise": (
            "TechNova Corp's infrastructure team reduced cloud costs by 34% using spot "
            "instances and autoscaling, saving approximately $8 million annually."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 9,
        "name": "bonus_policy",
        "query": "What is TechNova Corp's annual bonus structure?",
        "reference_answer": (
            "TechNova Corp employees are eligible for an annual performance bonus of up to "
            "15% of base salary, paid in February based on individual and company performance."
        ),
        "golden": (
            "TechNova Corp annual bonus policy: Eligible employees can receive a performance "
            "bonus of up to 15% of their annual base salary. Bonus amounts are determined by "
            "individual performance ratings and overall company goal attainment. Bonuses are "
            "paid in February for the prior calendar year."
        ),
        "redundants": [
            (
                "TechNova Corp pays annual performance bonuses of up to 15% of base salary "
                "in February, based on individual and company-wide performance metrics."
            ),
            (
                "TechNova's bonus program offers up to 15% of annual salary based on "
                "performance. Payouts occur each February for the preceding year."
            ),
            (
                "At TechNova Corp, the annual bonus can be as high as 15% of base salary. "
                "It is paid every February and depends on both individual and company results."
            ),
        ],
        "contradiction": (
            "TechNova Corp bonus policy: Employees may receive a discretionary bonus of up "
            "to 5% of salary. Bonuses are not guaranteed and are distributed at the CEO's "
            "discretion in Q1."
        ),
        "noise": (
            "TechNova Corp's zero-trust security architecture achieved SOC 2 Type II "
            "certification and ISO 27001 compliance, enabling expansion into regulated industries."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 10,
        "name": "expense_reimbursement",
        "query": "What is TechNova Corp's expense reimbursement process?",
        "reference_answer": (
            "TechNova Corp requires expense reports to be submitted within 60 days. "
            "Expenses over $500 require manager pre-approval."
        ),
        "golden": (
            "TechNova Corp expense reimbursement policy: Employees must submit expense "
            "reports within 60 days of incurring the expense. Receipts are required for all "
            "purchases over $25. Any single expense over $500 requires prior written approval "
            "from the employee's direct manager before the purchase is made."
        ),
        "redundants": [
            (
                "TechNova Corp reimburses business expenses submitted within 60 days. "
                "Receipts are needed for amounts over $25, and expenses above $500 need "
                "advance manager approval."
            ),
            (
                "Under TechNova's expense policy, reports must be filed within 60 days. "
                "Manager pre-approval is required for any single expense exceeding $500."
            ),
            (
                "TechNova Corp's reimbursement process: submit within 60 days with receipts "
                "for purchases over $25. Expenses exceeding $500 require manager sign-off first."
            ),
        ],
        "contradiction": (
            "TechNova Corp expense policy: Expenses must be submitted within 30 days. "
            "All expenses, regardless of amount, require manager approval before submission."
        ),
        "noise": (
            "TechNova Corp's product team shipped 47 new features in the last quarter, "
            "improving customer satisfaction scores by 12 points on the NPS survey."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 11,
        "name": "travel_policy",
        "query": "What class of air travel does TechNova Corp reimburse for business trips?",
        "reference_answer": (
            "TechNova Corp reimburses economy class for flights under 6 hours and business "
            "class for flights over 6 hours. Hotel stays are reimbursed up to $200 per night."
        ),
        "golden": (
            "TechNova Corp travel policy: For flights shorter than 6 hours, only economy "
            "class is reimbursable. Business class is approved for flights exceeding 6 hours. "
            "Hotel accommodations are reimbursed up to $200 per night. All travel must be "
            "booked through the company's approved travel portal."
        ),
        "redundants": [
            (
                "TechNova reimburses economy airfare for trips under 6 hours and business "
                "class for longer flights. Hotel reimbursement is capped at $200 per night."
            ),
            (
                "TechNova Corp travel rules: economy for flights under 6 hours, business "
                "class for over 6 hours, and hotels up to $200/night."
            ),
            (
                "For business travel, TechNova Corp covers economy class (flights < 6 hrs) "
                "or business class (flights > 6 hrs), plus hotels up to $200 per night."
            ),
        ],
        "contradiction": (
            "TechNova Corp travel policy: Economy class is required for all flights regardless "
            "of duration. Hotel reimbursement is limited to $150 per night."
        ),
        "noise": (
            "TechNova Corp's open-source contributions topped 200,000 commits last year, "
            "with engineers contributing to Kubernetes, PostgreSQL, and the Linux kernel."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 12,
        "name": "equipment_policy",
        "query": "What equipment does TechNova Corp provide to new employees?",
        "reference_answer": (
            "TechNova Corp provides a MacBook Pro to new employees, plus an $800 annual "
            "budget for peripherals and accessories."
        ),
        "golden": (
            "TechNova Corp equipment policy: All new employees receive a company-issued "
            "MacBook Pro (14-inch or 16-inch based on role). In addition, each employee "
            "receives an $800 annual budget for peripherals such as monitors, keyboards, "
            "mice, and headsets. The peripheral budget is non-transferable and expires "
            "December 31 each year."
        ),
        "redundants": [
            (
                "TechNova Corp gives new hires a MacBook Pro and $800/year to spend on "
                "accessories and peripherals like monitors and keyboards."
            ),
            (
                "New TechNova Corp employees get a MacBook Pro plus an $800 annual "
                "accessories budget covering monitors, keyboards, and other peripherals."
            ),
            (
                "TechNova Corp equipment: a MacBook Pro is issued on day one, and employees "
                "get $800 per year for peripheral purchases."
            ),
        ],
        "contradiction": (
            "TechNova Corp equipment policy: Employees choose between a MacBook Pro or "
            "a Dell XPS laptop. The peripheral budget is $300 per year, usable for "
            "pre-approved items only."
        ),
        "noise": (
            "TechNova Corp's latest platform update reduced API latency by 40%, bringing "
            "the p99 response time from 250ms to under 150ms globally."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 13,
        "name": "internet_stipend",
        "query": "Does TechNova Corp provide a home internet stipend for remote workers?",
        "reference_answer": (
            "Yes, TechNova Corp provides a $75 per month home internet stipend for all "
            "full-time employees who work remotely at least 1 day per week."
        ),
        "golden": (
            "TechNova Corp internet stipend: Full-time employees who work remotely at least "
            "one day per week are eligible for a $75 monthly home internet reimbursement. "
            "Employees submit a copy of their monthly internet bill for reimbursement through "
            "the expense portal."
        ),
        "redundants": [
            (
                "TechNova Corp reimburses $75 per month for home internet for employees "
                "who work remotely at least once a week."
            ),
            (
                "Remote-eligible TechNova Corp employees receive a $75/month internet "
                "stipend by submitting their internet bill through the expense system."
            ),
            (
                "TechNova Corp internet reimbursement: $75 monthly for full-time staff "
                "working from home at least one day per week."
            ),
        ],
        "contradiction": (
            "TechNova Corp does not offer a home internet stipend. Internet costs are "
            "considered a personal expense and are not reimbursable."
        ),
        "noise": (
            "TechNova Corp's diversity initiative resulted in 43% of 2024 engineering hires "
            "being from underrepresented groups, exceeding the company's stated 40% target."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 14,
        "name": "overtime_policy",
        "query": "How are overtime hours compensated at TechNova Corp?",
        "reference_answer": (
            "TechNova Corp pays non-exempt employees 1.5x their hourly rate for hours worked "
            "beyond 40 per week. Exempt employees receive compensatory time off."
        ),
        "golden": (
            "TechNova Corp overtime policy: Non-exempt (hourly) employees are paid at 1.5 "
            "times their standard hourly rate for all hours worked beyond 40 in a workweek, "
            "in compliance with the Fair Labor Standards Act. Exempt (salaried) employees do "
            "not receive overtime pay but may receive compensatory time off at their manager's "
            "discretion."
        ),
        "redundants": [
            (
                "TechNova Corp compensates non-exempt employees at 1.5x their regular pay "
                "for overtime hours (over 40/week). Exempt employees get comp time instead."
            ),
            (
                "For non-exempt TechNova Corp employees, overtime beyond 40 weekly hours "
                "is paid at time-and-a-half. Exempt salaried staff may receive compensatory time."
            ),
            (
                "TechNova overtime rules: hourly (non-exempt) workers earn 1.5x pay after "
                "40 hours/week; salaried (exempt) employees may be given comp time at manager discretion."
            ),
        ],
        "contradiction": (
            "TechNova Corp overtime policy: All employees, regardless of exempt status, "
            "receive 1.5x pay for hours over 40 per week. There is no compensatory time option."
        ),
        "noise": (
            "TechNova Corp's new AI coding assistant boosted engineering productivity by 25%, "
            "reducing average code review cycle time from 3 days to under 18 hours."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    # ── Multi-hop topics (15-19) ─────────────────────────────────────────────
    {
        "id": 15,
        "name": "total_compensation",
        "query": (
            "What is the total maximum compensation a TechNova Corp software engineer "
            "can receive, combining base salary and annual bonus?"
        ),
        "reference_answer": (
            "A TechNova Corp software engineer earns a base salary of up to $160,000. "
            "With the annual bonus of up to 15% of base, total maximum compensation "
            "is up to $184,000."
        ),
        "golden": (
            "TechNova Corp software engineer base salary range: Software engineers at "
            "TechNova Corp earn between $120,000 and $160,000 per year depending on level "
            "and experience. The exact salary is determined during the offer negotiation and "
            "annual review process."
        ),
        "redundants": [
            (
                "TechNova Corp engineer salaries range from $120k to $160k annually, "
                "varying by engineering level (L3 through L7) and years of experience."
            ),
            (
                "Software engineering compensation at TechNova Corp: base pay ranges "
                "from $120,000 for entry-level (L3) to $160,000 for senior (L6) engineers."
            ),
        ],
        "contradiction": (
            "TechNova Corp engineer salaries are standardized at $95,000 regardless of "
            "level or experience. All compensation increases happen through the bonus system."
        ),
        "noise": (
            "TechNova Corp's annual developer conference drew 8,000 attendees and featured "
            "120 technical sessions on cloud architecture, ML systems, and developer productivity."
        ),
        "part1": (
            "TechNova Corp software engineer base salary range: Software engineers at "
            "TechNova Corp earn between $120,000 and $160,000 per year depending on level "
            "and experience."
        ),
        "part2": (
            "TechNova Corp annual bonus policy: Eligible employees can receive a performance "
            "bonus of up to 15% of their annual base salary, paid in February for the prior year."
        ),
        "is_multihop": True,
    },
    {
        "id": 16,
        "name": "total_leave",
        "query": (
            "How many total paid days off (combining vacation and sick leave) does a "
            "TechNova Corp employee receive per year?"
        ),
        "reference_answer": (
            "TechNova Corp employees receive 15 vacation days plus 10 sick days, "
            "totaling 25 paid days off per year."
        ),
        "golden": (
            "TechNova Corp vacation entitlement: Full-time employees receive 15 paid "
            "vacation days per year. Employees with over 5 years of service receive 20 "
            "vacation days annually."
        ),
        "redundants": [
            (
                "TechNova Corp grants 15 annual vacation days to full-time employees, "
                "rising to 20 days for those with more than 5 years of service."
            ),
            (
                "TechNova Corp vacation days: 15 per year for all full-time staff, with "
                "a bonus of 5 extra days upon reaching 5 years of employment."
            ),
        ],
        "contradiction": (
            "TechNova Corp total leave: Employees receive 7 vacation days and 3 sick days "
            "annually, for a total of 10 paid days off per year."
        ),
        "noise": (
            "TechNova Corp's sustainability report showed a 28% reduction in carbon "
            "emissions compared to the prior year, attributed to renewable energy credits "
            "and server efficiency improvements."
        ),
        "part1": (
            "TechNova Corp vacation policy: All full-time employees receive 15 paid vacation "
            "days per year. After 5 years of service, this increases to 20 days annually."
        ),
        "part2": (
            "TechNova Corp sick leave policy: Full-time employees are entitled to 10 paid "
            "sick days per year, usable for personal illness or immediate family care."
        ),
        "is_multihop": True,
    },
    {
        "id": 17,
        "name": "software_stack",
        "query": (
            "What backend language and frontend framework does TechNova Corp use "
            "for its main product platform?"
        ),
        "reference_answer": (
            "TechNova Corp uses Python with FastAPI for the backend and React with "
            "TypeScript for the frontend of its main product platform."
        ),
        "golden": (
            "TechNova Corp backend architecture: The core product platform is built on "
            "Python using the FastAPI framework. Services are containerized with Docker "
            "and orchestrated via Kubernetes. The backend exposes RESTful and gRPC APIs."
        ),
        "redundants": [
            (
                "TechNova Corp's backend is Python-based, using FastAPI for the web framework. "
                "Services run in Docker containers managed by Kubernetes."
            ),
            (
                "The TechNova Corp platform backend uses Python and FastAPI, deployed "
                "in Kubernetes-managed Docker containers with REST and gRPC interfaces."
            ),
        ],
        "contradiction": (
            "TechNova Corp engineering stack: The platform is built on Java using Spring "
            "Boot for the backend and Angular for the frontend. Node.js is used for "
            "real-time services."
        ),
        "noise": (
            "TechNova Corp's customer success team achieved a 96% renewal rate for "
            "enterprise contracts, up from 89% the prior year following a new onboarding program."
        ),
        "part1": (
            "TechNova Corp backend architecture: The core product platform is built on "
            "Python using the FastAPI framework. Services are containerized with Docker "
            "and orchestrated via Kubernetes."
        ),
        "part2": (
            "TechNova Corp frontend architecture: The user interface is built with React "
            "and TypeScript. The design system uses Tailwind CSS, and state management "
            "is handled by Zustand."
        ),
        "is_multihop": True,
    },
    {
        "id": 18,
        "name": "data_infrastructure",
        "query": (
            "Where are TechNova Corp's primary and disaster recovery data centers located?"
        ),
        "reference_answer": (
            "TechNova Corp's primary data center is in AWS us-east-1 (Northern Virginia) "
            "and the disaster recovery data center is in GCP us-central1 (Iowa)."
        ),
        "golden": (
            "TechNova Corp primary infrastructure: All primary production workloads run "
            "in AWS us-east-1 (Northern Virginia). The region was chosen for its low latency "
            "to the East Coast customer base and proximity to financial industry clients."
        ),
        "redundants": [
            (
                "TechNova Corp primary data center: AWS us-east-1 in Northern Virginia. "
                "This region hosts all production databases, APIs, and compute workloads."
            ),
            (
                "Production infrastructure for TechNova Corp is hosted in AWS us-east-1 "
                "(Northern Virginia), selected for its reliability and customer proximity."
            ),
        ],
        "contradiction": (
            "TechNova Corp infrastructure is entirely hosted in a private on-premises "
            "data center in Austin, Texas. The company does not use public cloud services."
        ),
        "noise": (
            "TechNova Corp's annual employee survey showed 87% job satisfaction, with "
            "engineering, product, and design teams reporting the highest engagement scores."
        ),
        "part1": (
            "TechNova Corp primary data center: All production workloads run in AWS "
            "us-east-1 (Northern Virginia). This is the primary region for all customer-facing services."
        ),
        "part2": (
            "TechNova Corp disaster recovery: The DR environment is maintained in "
            "GCP us-central1 (Iowa), providing geographic redundancy. Failover can be "
            "completed within 15 minutes via automated runbooks."
        ),
        "is_multihop": True,
    },
    {
        "id": 19,
        "name": "sla_metrics",
        "query": (
            "What is TechNova Corp's API throughput capacity and uptime SLA?"
        ),
        "reference_answer": (
            "TechNova Corp's API platform handles 50,000 requests per second and "
            "guarantees 99.99% uptime in its enterprise SLA."
        ),
        "golden": (
            "TechNova Corp API throughput: The platform's API gateway is designed to "
            "handle up to 50,000 requests per second under peak load. Auto-scaling "
            "triggers at 70% capacity to ensure headroom for traffic spikes."
        ),
        "redundants": [
            (
                "TechNova Corp's API layer supports 50,000 requests per second, with "
                "autoscaling engaged at 70% capacity to manage traffic bursts."
            ),
            (
                "The TechNova Corp API gateway capacity is 50k RPS with auto-scaling "
                "configured to activate at 70% utilization."
            ),
        ],
        "contradiction": (
            "TechNova Corp's API platform is rated for 5,000 requests per second. "
            "The enterprise SLA guarantees 99.9% uptime."
        ),
        "noise": (
            "TechNova Corp's intern program accepted 200 students from 85 universities "
            "in 2024, with 78% receiving full-time offers upon graduation."
        ),
        "part1": (
            "TechNova Corp API throughput: The API gateway handles up to 50,000 requests "
            "per second under peak load, with autoscaling triggered at 70% capacity."
        ),
        "part2": (
            "TechNova Corp SLA commitments: Enterprise customers are guaranteed 99.99% "
            "uptime per calendar month. Downtime beyond this threshold triggers automatic "
            "service credits of 10% of monthly fees per 0.01% of excess downtime."
        ),
        "is_multihop": True,
    },
]


# ---------------------------------------------------------------------------
# 5 Medical-domain topics (hard task only — cross-domain generalization)
# ---------------------------------------------------------------------------

MEDICAL_TOPICS = [
    {
        "id": 20,
        "name": "drug_interaction",
        "query": "What is the mechanism behind the interaction between warfarin and NSAIDs?",
        "reference_answer": (
            "NSAIDs inhibit platelet aggregation and can damage the gastric mucosa, and when "
            "combined with warfarin's anticoagulant effect this significantly raises the risk "
            "of gastrointestinal bleeding. NSAIDs can also displace warfarin from plasma protein "
            "binding sites, increasing free warfarin concentration and INR."
        ),
        "golden": (
            "Drug interaction — warfarin and NSAIDs: NSAIDs inhibit COX-1-mediated thromboxane A2 "
            "production, impairing platelet aggregation. Concurrent use with warfarin, a vitamin K "
            "antagonist, markedly elevates bleeding risk, particularly GI haemorrhage. NSAIDs may "
            "also competitively displace warfarin from albumin binding sites, raising free warfarin "
            "plasma concentration and prolonging INR beyond the therapeutic range."
        ),
        "redundants": [
            (
                "Warfarin–NSAID interaction: NSAIDs reduce platelet function by blocking COX-1, and "
                "when combined with warfarin's blood-thinning action the additive antihaemostatic "
                "effect substantially increases the likelihood of serious bleeding, especially in the "
                "gastrointestinal tract. NSAIDs may displace warfarin from albumin, elevating INR."
            ),
            (
                "Clinical pharmacology note: co-prescribing NSAIDs with warfarin is a well-documented "
                "high-risk combination. The dual mechanism—platelet inhibition plus protein-binding "
                "displacement increasing free warfarin—raises GI bleed risk significantly compared "
                "with warfarin monotherapy."
            ),
            (
                "Pharmacokinetic interaction: NSAIDs compete with warfarin for plasma protein "
                "binding, freeing a greater fraction of warfarin and potentiating anticoagulation. "
                "Pharmacodynamically, NSAID-induced platelet dysfunction amplifies warfarin's "
                "haemostatic impairment, making concurrent use hazardous."
            ),
        ],
        "contradiction": (
            "Drug interaction note (outdated formulary): ibuprofen and warfarin are generally "
            "considered safe to co-administer at standard analgesic doses. No clinically significant "
            "INR elevation or bleeding risk has been observed in routine outpatient settings."
        ),
        "noise": (
            "The hospital pharmacy department completed its annual medication reconciliation audit, "
            "reviewing 3,400 patient medication lists across 12 wards for omission and duplication errors."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 21,
        "name": "amoxicillin_dosage",
        "query": "What is the standard adult dosage regimen for amoxicillin in treating community-acquired pneumonia?",
        "reference_answer": (
            "The standard adult dosage for amoxicillin in community-acquired pneumonia is 500 mg "
            "three times daily (every 8 hours) for 5–7 days for mild-to-moderate cases. Severe "
            "cases may require 1 g three times daily or IV amoxicillin-clavulanate."
        ),
        "golden": (
            "Amoxicillin dosing — community-acquired pneumonia (CAP): for mild-to-moderate CAP in "
            "adults, the recommended regimen is amoxicillin 500 mg orally every 8 hours for 5–7 days. "
            "For higher-severity outpatient CAP, 1 g every 8 hours is appropriate. Patients requiring "
            "hospitalisation should transition to IV beta-lactam therapy. Renal dose adjustment is "
            "required for eGFR below 30 mL/min."
        ),
        "redundants": [
            (
                "CAP treatment guideline: first-line antibiotic for low-severity CAP in adults without "
                "comorbidities is amoxicillin 500 mg three times daily for 5 to 7 days. Atypical "
                "cover with a macrolide should be added if atypical pathogens are suspected."
            ),
            (
                "Prescribing note — amoxicillin: standard adult dose for respiratory tract infection "
                "is 500 mg eight-hourly. Duration for pneumonia is typically one week. Dose escalation "
                "to 1 g TDS is used in more severe ambulatory presentations."
            ),
            (
                "Antimicrobial guideline summary: amoxicillin 500 mg orally every 8 hours for "
                "5–7 days remains the preferred oral regimen for outpatient mild-to-moderate "
                "community-acquired pneumonia in immunocompetent adults."
            ),
        ],
        "contradiction": (
            "CAP therapy note (prior edition): amoxicillin 250 mg twice daily for 3 days is "
            "sufficient for all ambulatory pneumonia cases, including those with moderate severity. "
            "Longer courses have not demonstrated superiority in recent trial data."
        ),
        "noise": (
            "The respiratory medicine department recorded a 12% reduction in hospital-acquired "
            "pneumonia rates following the introduction of a mandatory oral-care bundle across all ICU beds."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 22,
        "name": "informed_consent",
        "query": "What are the required elements of valid informed consent for a surgical procedure?",
        "reference_answer": (
            "Valid informed consent requires: disclosure of the nature and purpose of the procedure, "
            "material risks and benefits, reasonable alternatives, and likely consequences of refusal. "
            "The patient must have decision-making capacity, understand the information, and give "
            "consent voluntarily without coercion."
        ),
        "golden": (
            "Informed consent — required elements: a legally and ethically valid consent requires "
            "(1) disclosure of the procedure's nature, purpose, and expected outcomes; "
            "(2) material risks, including serious and common complications; "
            "(3) reasonable alternative treatments; "
            "(4) consequences of declining treatment; "
            "(5) patient decision-making capacity; "
            "(6) comprehension—information must be communicated in understandable terms; "
            "(7) voluntariness—consent must be free from coercion or undue influence."
        ),
        "redundants": [
            (
                "Consent law overview: informed consent is valid only when the clinician has disclosed "
                "the procedure's nature, significant risks, treatment alternatives, and the outcome "
                "of non-treatment. The patient must be competent, must understand the information, "
                "and must consent without pressure."
            ),
            (
                "Medical ethics guidance: the seven core elements of informed consent are nature of "
                "the intervention, its purpose, material risks, benefits, alternatives, consequences "
                "of refusal, and patient autonomy. Capacity and voluntariness are prerequisites."
            ),
            (
                "Surgical consent checklist: before operating, surgeons must confirm the patient "
                "has been informed of procedure details, significant complications, non-surgical "
                "options, and consequences of no treatment. Patient capacity and voluntary agreement "
                "must be documented."
            ),
        ],
        "contradiction": (
            "Consent note (outdated guidance): obtaining a patient signature on the consent form "
            "is legally sufficient evidence of informed consent. Verbal explanation of risks is "
            "optional if the form lists standard complications of the procedure."
        ),
        "noise": (
            "The hospital's patient experience survey showed an 88% satisfaction score, with "
            "communication of care plans rated highest among inpatients across all wards."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 23,
        "name": "hipaa_breach",
        "query": "What is the HIPAA breach notification timeline for covered entities?",
        "reference_answer": (
            "Under HIPAA, covered entities must notify affected individuals within 60 days of "
            "discovering a breach. If the breach affects 500 or more residents of a state, the "
            "entity must also notify prominent media outlets and the HHS Secretary without "
            "unreasonable delay and within 60 days. Smaller breaches must be logged and reported "
            "to HHS annually."
        ),
        "golden": (
            "HIPAA Breach Notification Rule — covered entity obligations: upon discovery of a "
            "breach of unsecured PHI, covered entities must: "
            "(1) notify each affected individual no later than 60 calendar days after breach discovery; "
            "(2) notify the HHS Secretary without unreasonable delay and within 60 days; "
            "(3) for breaches affecting 500+ residents of a state or jurisdiction, notify prominent "
            "local media within 60 days. Breaches affecting fewer than 500 individuals must be "
            "logged and submitted to HHS in an annual report."
        ),
        "redundants": [
            (
                "HIPAA breach rule summary: covered entities have 60 days from breach discovery to "
                "notify affected individuals. Large breaches (500+ state residents) also trigger "
                "media notification and immediate HHS reporting. Small breaches are reported annually."
            ),
            (
                "Breach notification requirement: the 60-day clock starts on the date the covered "
                "entity becomes aware of the breach. Notifications to patients, HHS, and media "
                "(for large breaches) must all occur within this window."
            ),
            (
                "Healthcare compliance note: HIPAA mandates individual notifications within 60 days "
                "of breach discovery for all covered entities. Breaches involving 500 or more "
                "individuals in a single state require concurrent media notice and expedited HHS filing."
            ),
        ],
        "contradiction": (
            "HIPAA compliance memo (prior guidance): covered entities have up to 120 days to notify "
            "affected individuals following a PHI breach. Media notification is only required if "
            "the breach affects more than 1,000 individuals in a single state."
        ),
        "noise": (
            "The health information management department completed migration of 200,000 paper "
            "records to the new electronic health record system, reducing retrieval time from 4 hours to under 3 minutes."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 24,
        "name": "clinical_trial_phases",
        "query": "What distinguishes a Phase II clinical trial from a Phase III clinical trial?",
        "reference_answer": (
            "Phase II trials primarily assess efficacy signals and dosing in a small patient group "
            "(typically 100–300 participants) to determine whether the intervention works well "
            "enough to warrant a larger study. Phase III trials are large, often multi-centre "
            "randomised controlled trials (hundreds to thousands of participants) designed to "
            "confirm efficacy, monitor adverse reactions, and compare against standard treatments "
            "before regulatory submission."
        ),
        "golden": (
            "Clinical trial phases — II vs III: Phase II trials enrol 100–300 participants and "
            "are designed to generate preliminary evidence of efficacy, identify the optimal dose "
            "range, and evaluate short-term safety in the target population. If Phase II results "
            "are promising, Phase III trials are initiated with 300–3,000+ participants across "
            "multiple sites. Phase III uses randomisation and control arms to rigorously confirm "
            "efficacy, quantify adverse event rates, and support regulatory approval applications."
        ),
        "redundants": [
            (
                "Drug development overview: Phase II studies (100–300 subjects) test whether a drug "
                "shows the expected therapeutic effect and refine dosing. Phase III studies enrol "
                "hundreds to thousands of patients in randomised controlled designs to confirm "
                "efficacy and safety before a regulatory submission."
            ),
            (
                "Trial phase distinction: Phase II focuses on efficacy signals and dose optimisation "
                "in small cohorts. Phase III is the pivotal confirmatory stage, using large "
                "randomised samples to establish the risk-benefit profile needed for market approval."
            ),
            (
                "Regulatory science note: progression from Phase II to Phase III requires evidence "
                "of biological activity and an acceptable safety profile. Phase III randomised "
                "trials provide the statistical power needed for regulatory decision-making and "
                "post-market surveillance design."
            ),
        ],
        "contradiction": (
            "Trial design note (misstatement): Phase II and Phase III trials are largely "
            "interchangeable in design. Both use randomisation, placebo controls, and large "
            "sample sizes, and either may be submitted directly to regulators as pivotal evidence."
        ),
        "noise": (
            "The clinical research office enrolled its 10,000th trial participant, reflecting a "
            "threefold increase in research capacity following the opening of the dedicated clinical trials unit."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
]


# ---------------------------------------------------------------------------
# PyTorch / Meta / Scalar topics (hard task only — cross-domain generalization)
# ---------------------------------------------------------------------------

PYTORCH_TOPICS = [
    {
        "id": 25,
        "name": "torch_inference_mode",
        "query": "What is the difference between torch.no_grad() and torch.inference_mode() in PyTorch?",
        "reference_answer": (
            "torch.inference_mode() is a stronger context manager than torch.no_grad(). Both disable "
            "gradient computation, but inference_mode also disables view tracking and version counting, "
            "making it faster. Tensors created inside inference_mode cannot be used in autograd graphs "
            "later, while no_grad() tensors can. inference_mode is preferred for inference-only code."
        ),
        "golden": (
            "PyTorch inference context managers: torch.no_grad() disables gradient computation within "
            "its scope, reducing memory usage and speeding up operations that do not require gradients. "
            "torch.inference_mode() is a superset — it additionally disables PyTorch's version counter "
            "and view tracking, enabling further performance optimizations. Tensors created in "
            "inference_mode are marked as not requiring gradients and cannot be used in autograd graphs "
            "after the context exits. For pure inference workloads, inference_mode is recommended over "
            "no_grad()."
        ),
        "redundants": [
            (
                "PyTorch gradient context: torch.inference_mode() should be preferred over "
                "torch.no_grad() for inference code. While both suppress gradient tracking, "
                "inference_mode goes further by disabling version counters and view tracking, "
                "yielding additional speed gains. Tensors produced inside inference_mode are not "
                "autograd-compatible after the block ends."
            ),
            (
                "PyTorch docs note: inference_mode is a stricter, faster alternative to no_grad. "
                "The key difference is that inference_mode tensors cannot participate in autograd "
                "computation graphs, whereas no_grad tensors remain autograd-compatible. Use "
                "inference_mode() for model evaluation loops and inference pipelines."
            ),
            (
                "Performance guide — PyTorch context managers: using torch.inference_mode() in "
                "evaluation and inference code is preferred over torch.no_grad() because it disables "
                "more internal tracking mechanisms. The trade-off is that tensors inside "
                "inference_mode cannot later be fed into autograd-tracked operations."
            ),
        ],
        "contradiction": (
            "PyTorch API note (outdated): torch.no_grad() and torch.inference_mode() are functionally "
            "identical. Both disable gradient computation and version tracking. inference_mode offers "
            "no performance advantage and can be used interchangeably with no_grad() in all contexts "
            "including training loops."
        ),
        "noise": (
            "The PyTorch 2.0 release exceeded 10 million monthly downloads on PyPI, making it one "
            "of the most widely adopted deep learning frameworks globally."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 26,
        "name": "fsdp_vs_ddp",
        "query": (
            "When should you choose FSDP over DDP for distributed PyTorch training, "
            "and what are the memory implications of each strategy?"
        ),
        "reference_answer": (
            "Choose FSDP (Fully Sharded Data Parallel) when your model is too large to fit in a "
            "single GPU's memory — FSDP shards model parameters, gradients, and optimizer states "
            "across all GPUs, so each GPU holds only a fraction of the model. DDP (Distributed Data "
            "Parallel) replicates the full model on every GPU and synchronizes gradients after each "
            "backward pass. DDP has lower communication overhead when the model fits in GPU memory; "
            "FSDP is necessary for models that exceed single-GPU VRAM."
        ),
        "golden": (
            "PyTorch distributed training — FSDP overview: Fully Sharded Data Parallel (FSDP) shards "
            "model parameters, gradients, and optimizer states across all participating GPUs. Each GPU "
            "stores only 1/N of the total model weight, where N is the number of GPUs. During the "
            "forward and backward passes, FSDP gathers the required shards via all-gather collectives "
            "before each layer and discards them afterward. This enables training of models that far "
            "exceed single-GPU VRAM (e.g., 70B+ parameter models on A100 clusters)."
        ),
        "redundants": [
            (
                "FSDP documentation: FSDP partitions model parameters, gradients, and optimizer states "
                "across GPUs. A 70B parameter model in BF16 requires ~140GB — far beyond a single "
                "A100's 80GB. With 4 GPUs and FSDP, each GPU holds ~35GB of parameters. FSDP uses "
                "all-gather before each layer and reduce-scatter after backward to synchronize gradients."
            ),
            (
                "Distributed training guide: use FSDP when your model does not fit on a single device. "
                "FSDP shards all tensors (parameters, gradients, optimizer states) evenly across the "
                "process group. This reduces per-GPU memory to approximately "
                "(total_model_size + optimizer_states) / num_gpus, enabling large model training."
            ),
        ],
        "contradiction": (
            "PyTorch distributed note (outdated): FSDP and DDP have identical memory footprints per "
            "GPU. Both strategies replicate the full model on each device and synchronize gradients "
            "after each backward pass. FSDP only differs in its gradient bucketing strategy, not in "
            "parameter sharding."
        ),
        "noise": (
            "Meta AI's distributed systems team released a blog post detailing the infrastructure "
            "used to train LLaMA 3, including custom interconnect topology and gradient checkpointing."
        ),
        "part1": (
            "PyTorch DDP — Distributed Data Parallel: DDP replicates the full model on every GPU in "
            "the process group. Each GPU processes a different mini-batch, computes gradients, and "
            "DDP performs an all-reduce to synchronize gradients across all replicas. DDP is efficient "
            "when the model fits in a single GPU's memory and is the simplest strategy for data-parallel "
            "training. Communication overhead is proportional to model size."
        ),
        "part2": (
            "PyTorch FSDP — Fully Sharded Data Parallel: FSDP shards model parameters, gradients, and "
            "optimizer states across all GPUs. Each device holds only 1/N of total parameters. FSDP is "
            "required when model size exceeds single-GPU VRAM. It introduces all-gather communication "
            "before each layer's forward/backward pass, which increases communication volume but reduces "
            "memory pressure dramatically compared to DDP."
        ),
        "is_multihop": True,
    },
    {
        "id": 27,
        "name": "torch_compile",
        "query": "What does torch.compile() do in PyTorch 2.0 and which compilation backend does it use by default?",
        "reference_answer": (
            "torch.compile() uses TorchDynamo to capture the model's computation graph by inspecting "
            "Python bytecode at runtime, then compiles it using TorchInductor (the default backend). "
            "TorchInductor generates optimized Triton kernels for GPU execution or C++/OpenMP code for "
            "CPU. This can provide 2-10x speedups on training workloads by fusing operations and "
            "reducing Python overhead."
        ),
        "golden": (
            "torch.compile() — PyTorch 2.0 compilation: torch.compile() wraps a model or function and "
            "compiles it for faster execution. Internally it uses TorchDynamo, which captures the Python "
            "computation graph via bytecode analysis without requiring model rewrites. The captured graph "
            "is passed to a backend compiler; the default backend is TorchInductor, which generates "
            "Triton kernels for CUDA GPUs and C++/OpenMP code for CPU. torch.compile() supports three "
            "modes: 'default' (balanced), 'reduce-overhead' (minimizes CUDA launch overhead via "
            "CUDAGraphs), and 'max-autotune' (longest compile time, best runtime performance)."
        ),
        "redundants": [
            (
                "PyTorch compile guide: torch.compile() is the recommended way to accelerate PyTorch "
                "models in version 2.0+. It uses TorchDynamo for graph capture and TorchInductor as "
                "the default backend. TorchInductor produces fused Triton kernels on GPU, reducing "
                "kernel launch overhead and improving memory bandwidth utilization. Typical speedups "
                "range from 30% to 3x on standard training benchmarks."
            ),
            (
                "PyTorch 2.0 release notes: torch.compile() replaces torch.jit.script() as the "
                "preferred compilation path. The compilation stack (TorchDynamo → AOT Autograd → "
                "TorchInductor) enables graph-level optimizations without requiring static graphs. "
                "Dynamic shapes are supported with graph re-compilation on shape changes."
            ),
            (
                "Performance optimization — torch.compile: wrapping a model with torch.compile() "
                "invokes TorchDynamo to trace the computation graph and TorchInductor to generate "
                "hardware-optimized kernels. The 'max-autotune' mode uses kernel autotuning for "
                "best performance at the cost of longer compile time on first run."
            ),
        ],
        "contradiction": (
            "PyTorch compilation note (outdated): torch.compile() uses TorchScript as its default "
            "compilation backend. The compiled model is equivalent to torch.jit.trace() output and "
            "requires all inputs to have static shapes. Dynamic shapes are not supported and will "
            "cause recompilation failures."
        ),
        "noise": (
            "PyTorch 2.0 was officially released in March 2023 and set a new benchmark for eager mode "
            "performance, topping multiple MLPerf inference benchmarks in its release quarter."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 28,
        "name": "scalar_kernel_fusion",
        "query": "How does Scalar's kernel fusion approach reduce memory bandwidth overhead in GPU computation?",
        "reference_answer": (
            "Scalar fuses multiple element-wise and reduction operations into a single GPU kernel, "
            "eliminating intermediate memory reads and writes between operations. Instead of writing "
            "activation tensors to HBM between each operation, Scalar keeps intermediate values in GPU "
            "registers and shared memory. This reduces HBM bandwidth pressure by up to 5-10x for "
            "memory-bound workloads like attention, layer norm, and activation functions."
        ),
        "golden": (
            "Scalar kernel fusion — technical overview: Scalar's compiler analyzes a computation graph "
            "and fuses sequences of compatible operations into a single GPU kernel. Without fusion, each "
            "operation (e.g., linear → ReLU → dropout) launches a separate kernel that writes its output "
            "to HBM (high-bandwidth memory) and the next kernel reads it back. Scalar eliminates these "
            "intermediate HBM reads and writes by keeping values in on-chip GPU memory (registers and "
            "shared memory) across the entire fused operation. For memory-bandwidth-bound workloads — "
            "which includes most transformer building blocks — this fusion can reduce wall-clock time by "
            "3-10x."
        ),
        "redundants": [
            (
                "Scalar execution model: the core performance advantage of Scalar is operator fusion. "
                "By combining multiple GPU operations into one kernel, Scalar avoids redundant HBM "
                "traffic. A fused attention kernel avoids writing the attention score matrix to global "
                "memory entirely, keeping it in SRAM. This is equivalent to FlashAttention's approach "
                "but applied as a general compiler pass across all eligible operation sequences."
            ),
            (
                "Scalar optimization strategy: Scalar's MLIR-based compiler performs vertical fusion, "
                "merging chains of pointwise, reduction, and broadcast operations into single kernels. "
                "Memory-bound operations such as layer normalization, softmax, and element-wise "
                "activations benefit most from fusion because their compute intensity is low and HBM "
                "bandwidth is the bottleneck."
            ),
            (
                "Scalar architecture note: Scalar uses a single-pass kernel execution model where the "
                "compiler generates fused kernels that process data entirely within GPU on-chip memory. "
                "The elimination of intermediate HBM writes is critical for transformer inference, where "
                "attention, FFN activations, and layer norm are heavily memory-bandwidth-limited."
            ),
        ],
        "contradiction": (
            "Scalar execution note (outdated beta): Scalar launches one GPU kernel per operation, "
            "consistent with standard CUDA execution. Each operation reads inputs from and writes "
            "outputs to device memory (HBM). Kernel fusion is not applied in the current release; "
            "performance gains come from optimized memory access patterns within individual kernels."
        ),
        "noise": (
            "Scalar closed a Series A funding round with participation from infrastructure-focused "
            "venture capital firms, citing growing demand for high-performance numerical computing."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
    {
        "id": 29,
        "name": "llama3_memory",
        "query": "What are the GPU memory requirements and native context length for Meta's LLaMA 3 70B model?",
        "reference_answer": (
            "LLaMA 3 70B requires approximately 140GB of GPU memory for BF16 inference (2 bytes × "
            "70 billion parameters). A practical deployment uses two A100 80GB GPUs or four A6000 "
            "48GB GPUs for tensor-parallel inference. The native context length is 8,192 tokens, "
            "extendable to 128K tokens with Meta's LLaMA 3.1 70B long-context variant."
        ),
        "golden": (
            "Meta LLaMA 3 70B — deployment specifications: LLaMA 3 70B has 70 billion parameters. "
            "In BF16 precision (2 bytes per parameter), the model weights alone require ~140GB of GPU "
            "memory, excluding KV cache and activation memory. A standard deployment uses tensor "
            "parallelism across 2× NVIDIA A100 80GB GPUs (160GB total) or 4× A6000 48GB GPUs "
            "(192GB total). The base LLaMA 3 70B model supports a native context window of 8,192 "
            "tokens. LLaMA 3.1 70B extends this to 128K tokens using RoPE scaling and grouped query "
            "attention optimized for long contexts."
        ),
        "redundants": [
            (
                "LLaMA 3 70B memory guide: at BF16 precision, 70B parameters × 2 bytes = ~140GB "
                "minimum VRAM for weights. For inference serving, 2× A100 80GB is the typical "
                "configuration, providing headroom for the KV cache. The model's native sequence "
                "length is 8,192 tokens; the 3.1 variant extends to 128K tokens."
            ),
            (
                "Meta AI model card — LLaMA 3 70B: the 70B parameter model requires two A100 80GB "
                "GPUs for BF16 inference in standard configurations. Context length is 8,192 tokens "
                "for the base release. The LLaMA 3.1 series adds long-context support up to 128K "
                "tokens through architectural improvements including grouped query attention and "
                "updated RoPE frequency settings."
            ),
            (
                "Deployment guide — large language models: LLaMA 3 70B in BF16 needs ~140GB GPU "
                "memory for model weights. Adding the KV cache for a 4K context batch adds roughly "
                "10GB. Practical minimum: 2× A100-80GB. For long contexts (128K) with LLaMA 3.1, "
                "4-8× A100 GPUs are recommended depending on batch size."
            ),
        ],
        "contradiction": (
            "LLaMA deployment note (prior release): LLaMA 2 70B requires 35GB of GPU memory in "
            "4-bit quantization and fits on a single A100 80GB GPU. The model supports a context "
            "window of 4,096 tokens and does not support extension beyond this limit without "
            "fine-tuning on longer sequences."
        ),
        "noise": (
            "Meta AI open-sourced LLaMA 3 under a community license permitting commercial use for "
            "most deployments, with restrictions for services exceeding 700 million monthly active users."
        ),
        "part1": None,
        "part2": None,
        "is_multihop": False,
    },
]


# ---------------------------------------------------------------------------
# Document and corpus builders
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Rough token estimate: words * 1.3, minimum 10."""
    return max(10, int(len(text.split()) * 1.3))


def _build_doc(
    doc_id: int,
    content: str,
    source: str,
    doc_type: str,
    pattern: str,
    *,
    topic_id: int,
    support_type: str,
    fact_group: str = "policy",
    is_current: bool = True,
) -> dict:
    return {
        "doc_id": doc_id,
        "content": content,
        "source": source,
        "doc_type": doc_type,
        "pattern": pattern,
        "token_count": _token_count(content),
        "topic_id": topic_id,
        "support_type": support_type,
        "fact_group": fact_group,
        "is_current": is_current,
    }


# ---------------------------------------------------------------------------
# Per-topic required fact groups (specific, not generic "policy")
# Agents must cover these groups to achieve full evidence_recall.
# Golden/redundant chunks are tagged with the matching group in _build_doc calls.
# ---------------------------------------------------------------------------

_REQUIRED_FACT_GROUPS: dict[str, list] = {
    "vacation_policy":      ["vacation_entitlement"],
    "sick_leave":           ["sick_days_count"],
    "remote_work":          ["remote_days_allowed"],
    "health_benefits":      ["health_coverage_terms"],
    "401k_policy":          ["match_percentage"],
    "parental_leave":       ["caregiver_weeks"],
    "performance_review":   ["review_timing"],
    "learning_budget":      ["development_amount"],
    "stock_options":        ["vesting_schedule"],
    "bonus_policy":         ["bonus_ceiling"],
    "expense_reimbursement":["submission_window"],
    "travel_policy":        ["flight_class_rule"],
    "equipment_policy":     ["equipment_budget"],
    "internet_stipend":     ["stipend_amount"],
    "overtime_policy":      ["overtime_rate"],
    # multihop — two groups must both be covered
    "total_compensation":   ["salary_range", "bonus_percentage"],
    "total_leave":          ["vacation_days", "sick_days"],
    "software_stack":       ["backend_stack", "frontend_stack"],
    "data_infrastructure":  ["primary_dc", "dr_dc"],
    "sla_metrics":          ["api_throughput", "uptime_sla"],
    # medical domain (hard only)
    "drug_interaction":     ["interaction_mechanism"],
    "amoxicillin_dosage":   ["dosage_regimen"],
    "informed_consent":     ["consent_elements"],
    "hipaa_breach":         ["notification_timeline"],
    "clinical_trial_phases":["trial_phase_distinction"],
    # PyTorch / Meta / Scalar domain (hard only)
    "torch_inference_mode": ["inference_context_difference"],
    "fsdp_vs_ddp":          ["ddp_memory_model", "fsdp_memory_model"],  # multihop
    "torch_compile":        ["compilation_backend"],
    "scalar_kernel_fusion": ["fusion_mechanism"],
    "llama3_memory":        ["memory_requirement"],
}

# ---------------------------------------------------------------------------
# Per-topic adversarial summaries — unique misleading text per topic.
# Each sounds plausible but contains wrong specifics or outdated figures.
# ---------------------------------------------------------------------------

_ADVERSARIAL_TEXT: dict[str, str] = {
    "vacation_policy": (
        "TechNova Corp HR archive — vacation accrual memo (superseded): at initial hire, employees "
        "begin accruing leave at 1 day per completed month of service, capped at a legacy ceiling "
        "of 12 days in year one. Tenure adjustments were under review at the time of this memo."
    ),
    "sick_leave": (
        "TechNova Corp HR compliance note (prior edition): a draft policy limited sick leave usage "
        "to the employee's own medical condition only, capped at two consecutive days before a "
        "medical certificate was required. This draft was not adopted; refer to the active policy."
    ),
    "remote_work": (
        "TechNova Corp flexible work pilot — final report (archived): the 2021 pilot permitted "
        "eligible senior staff to work remotely five days per week. The pilot concluded and standard "
        "hybrid arrangements now apply; consult the current approved policy for day limits."
    ),
    "health_benefits": (
        "TechNova Corp benefits committee proposal (not adopted): a cost-sharing model was modelled "
        "in which employees would contribute $200 per month toward their individual health premium. "
        "The proposal was rejected; current benefits carry no employee premium."
    ),
    "401k_policy": (
        "TechNova Corp retirement plan — early summary sheet (superseded): the initial plan design "
        "described a tiered match reaching 4% of salary after two years of service. Final plan terms "
        "differ materially; the current plan document governs all contributions."
    ),
    "parental_leave": (
        "TechNova Corp statutory compliance memo (archived): a baseline of six weeks paid leave for "
        "all new parents was noted as the minimum statutory requirement at the time of drafting. "
        "The approved leave policy now provides significantly more generous entitlements."
    ),
    "performance_review": (
        "TechNova Corp people ops pilot note (archived): a quarterly check-in model was trialled, "
        "during which formal ratings and compensation recommendations were submitted each quarter. "
        "The pilot has concluded; the current annual cycle is the approved practice."
    ),
    "learning_budget": (
        "TechNova Corp L&D programme — seed budget proposal (not adopted): the original proposal "
        "allocated $500 per employee per year for online courses only. The approved programme "
        "differs in amount and eligible expense categories."
    ),
    "stock_options": (
        "TechNova Corp equity plan — retention grant memo (archived): accelerated two-year vesting "
        "was modelled for a proposed retention grant class. Standard option grants follow the "
        "full vesting schedule; the equity plan document governs all grants."
    ),
    "bonus_policy": (
        "TechNova Corp compensation planning note (prior fiscal year): an internal projection "
        "modelled a discretionary bonus pool of up to 8% of base salary under conservative "
        "performance assumptions. Actual eligibility and percentages are defined in active policy."
    ),
    "expense_reimbursement": (
        "TechNova Corp finance policy — legacy version: an older policy edition required expense "
        "submission within 30 days and manager approval for any amount over $200. These thresholds "
        "have since been updated; the current policy document applies."
    ),
    "travel_policy": (
        "TechNova Corp travel guidelines — pre-2022 edition: legacy guidelines approved business "
        "class for all transatlantic travel regardless of flight duration and capped hotel "
        "reimbursement at $150 per night. These rates and criteria have been revised."
    ),
    "equipment_policy": (
        "TechNova Corp IT refresh memo (2022 cycle): the 2022 hardware refresh allocated a $500 "
        "peripherals budget distributed biannually. Current allocations, refresh cadence, and "
        "approved device categories are documented in the active IT policy."
    ),
    "internet_stipend": (
        "TechNova Corp remote support proposal (early draft): an initial proposal discussed a "
        "$40 per month internet reimbursement limited to employees in certain designated regions. "
        "The approved stipend amount and full eligibility criteria are in the current policy."
    ),
    "overtime_policy": (
        "TechNova Corp HR compliance note — pre-reclassification: a compensation memo from before "
        "the FLSA reclassification review described overtime for reclassified employees at straight "
        "time only. Current overtime treatment follows the active policy and applicable law."
    ),
    "total_compensation": (
        "TechNova Corp salary benchmarking summary (archived): a compensation benchmarking exercise "
        "used a market reference range of $90,000–$130,000 for engineering roles. Actual current "
        "salary bands and bonus eligibility are in the live compensation framework."
    ),
    "total_leave": (
        "TechNova Corp leave consolidation memo (prior review): an internal review considered "
        "merging vacation and sick leave into a single PTO bank of 18 days total. The proposal "
        "was not adopted; vacation and sick leave remain separate entitlements."
    ),
    "software_stack": (
        "TechNova Corp architecture review board note (archived): a 2020 RFC proposed migrating "
        "the backend to Node.js and the frontend to Vue.js. The RFC was not approved; current "
        "technology choices are documented in the active architecture decision records."
    ),
    "data_infrastructure": (
        "TechNova Corp infrastructure planning note (archived): an early cloud strategy memo "
        "described a single-region deployment in AWS us-west-2 with no dedicated DR environment. "
        "Current infrastructure spans multiple regions; refer to the active runbook."
    ),
    "sla_metrics": (
        "TechNova Corp capacity planning memo (prior review): an earlier capacity model rated the "
        "API gateway at 10,000 requests per second with a 99.9% uptime commitment. Current "
        "published SLA terms and capacity figures supersede this estimate."
    ),
    # PyTorch / Meta / Scalar domain
    "torch_inference_mode": (
        "PyTorch API comparison (prior version): torch.no_grad() was the only recommended context "
        "manager for inference. torch.inference_mode() was added as an alias in 1.9 but is not "
        "considered production-ready for all use cases, particularly those involving autograd hooks. "
        "Both context managers have identical performance characteristics."
    ),
    "fsdp_vs_ddp": (
        "PyTorch distributed training guide (archived): FSDP is a research prototype not recommended "
        "for production workloads as of PyTorch 1.12. DDP is the only production-supported distributed "
        "strategy. FSDP's gradient sharding may cause numerical instability in mixed-precision training "
        "and does not improve memory efficiency compared to DDP."
    ),
    "torch_compile": (
        "PyTorch 2.0 beta note: torch.compile() has known compatibility issues with dynamic control "
        "flow and custom autograd functions. For production, torch.jit.script() remains the recommended "
        "path. TorchScript is the default compilation backend; TorchInductor support is experimental."
    ),
    "scalar_kernel_fusion": (
        "Scalar technical preview note (alpha release): kernel fusion in Scalar is currently applied "
        "only to pointwise operations. Reductions and attention mechanisms are not fused in the current "
        "version. Performance gains are limited to elementwise activation functions. HBM traffic "
        "reduction has not been demonstrated on transformer workloads."
    ),
    "llama3_memory": (
        "LLaMA deployment FAQ (prior version): LLaMA 2 70B can be served from a single A100 80GB GPU "
        "using 4-bit GPTQ quantization with minimal quality degradation. Context length is limited to "
        "4,096 tokens in this configuration, and extension to 128K tokens is not supported without "
        "architectural changes."
    ),
    # medical domain
    "drug_interaction": (
        "Drug interaction archive — prior formulary edition: ibuprofen was listed as low-risk at "
        "standard OTC doses alongside anticoagulants, with routine INR monitoring described as "
        "optional. Current interaction classification has been updated; consult the active formulary."
    ),
    "amoxicillin_dosage": (
        "Hospital formulary archive (prior edition): amoxicillin 250 mg three times daily was "
        "listed as the standard adult dose for respiratory tract infections. Current guidelines "
        "specify higher doses for pneumonia; the active formulary should be referenced."
    ),
    "informed_consent": (
        "Consent training module archive: a legacy module described three elements — purpose, "
        "procedure, and voluntary participation — as the minimum consent standard. Current "
        "regulatory requirements specify nine mandatory elements under 21 CFR 50.25."
    ),
    "hipaa_breach": (
        "HIPAA compliance training archive: an older training document described a 90-day "
        "notification window and characterised media notification as discretionary for large "
        "breaches. Current regulatory requirements set a 60-day deadline for all covered entities."
    ),
    "clinical_trial_phases": (
        "Clinical development briefing (internal, archived): an informal summary described Phase II "
        "as requiring 500+ participants and Phase III as a confirmatory study of any scale. "
        "ICH E8 and FDA guidance define the formal stage criteria; consult those documents."
    ),
}


def _query_entry(topic: dict, phrasing: str = "default") -> dict:
    query = topic["query"]
    if phrasing == "ops":
        query = f"Using the current TechNova documentation, {query[0].lower()}{query[1:]}"
    elif phrasing == "audit":
        query = f"Based on the latest approved policy, {query[0].lower()}{query[1:]}"

    # Use per-topic specific fact groups instead of generic "policy"
    required_fact_groups = _REQUIRED_FACT_GROUPS.get(topic["name"], ["policy"])

    return {
        "query_id": topic["id"],
        "query": query,
        "reference_answer": topic["reference_answer"],
        "required_fact_groups": required_fact_groups,
    }


def _adversarial_summary(topic: dict) -> str:
    """Return a unique, per-topic misleading summary — not a generic template."""
    return _ADVERSARIAL_TEXT.get(
        topic["name"],
        (
            f"TechNova Corp {topic['name']} archive: an older document version discusses this "
            "topic but contains figures from a prior policy cycle. The current approved document governs."
        ),
    )


def _generate_easy_corpus(rng: random.Random) -> dict:
    """50 documents, FAQ-style, redundancy patterns only.
    Golden and redundant chunks tagged with topic-specific fact_group
    so evidence_recall is meaningful (not a constant 'policy' group).
    """
    documents = []
    doc_id = 0

    # 20 golden + 20 redundant (first redundant only) = 40 docs
    for t in TOPIC_DEFINITIONS:
        fg = _REQUIRED_FACT_GROUPS.get(t["name"], ["policy"])[0]
        source = f"faq_{t['name']}.txt"
        documents.append(
            _build_doc(
                doc_id, t["golden"], source, "faq", "golden",
                topic_id=t["id"], support_type="gold", fact_group=fg,
            )
        )
        doc_id += 1
        documents.append(
            _build_doc(
                doc_id, t["redundants"][0], f"faq_{t['name']}_v1.txt", "faq", "redundant",
                topic_id=t["id"], support_type="support", fact_group=fg,
            )
        )
        doc_id += 1

    # 10 noise documents (unrelated to query topics — fact_group="noise")
    for t in TOPIC_DEFINITIONS[:10]:
        documents.append(
            _build_doc(
                doc_id, t["noise"], f"news_{t['name']}.txt", "news", "noise",
                topic_id=t["id"], support_type="noise", fact_group="noise",
            )
        )
        doc_id += 1

    queries = [_query_entry(t) for t in TOPIC_DEFINITIONS[:15]]

    return {"documents": documents, "queries": queries}


def _generate_medium_corpus(rng: random.Random) -> dict:
    """200 documents: policy + technical + news, redundancy + noise + contradictions."""
    documents = []
    doc_id = 0

    for t in TOPIC_DEFINITIONS:
        source_base = t["name"]
        doc_type = "policy" if not t["is_multihop"] else "technical"
        fgs = _REQUIRED_FACT_GROUPS.get(t["name"], ["policy"])
        fg_primary = fgs[0]

        documents.append(
            _build_doc(
                doc_id,
                t["golden"],
                f"{source_base}_policy.txt",
                doc_type,
                "golden",
                topic_id=t["id"],
                support_type="gold",
                fact_group=fg_primary,
            )
        )
        doc_id += 1

        for i, red in enumerate(t["redundants"]):
            documents.append(
                _build_doc(
                    doc_id,
                    red,
                    f"{source_base}_ref_{i}.txt",
                    doc_type,
                    "redundant",
                    topic_id=t["id"],
                    support_type="support",
                    fact_group=fg_primary,
                )
            )
            doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                t["contradiction"],
                f"{source_base}_old.txt",
                doc_type,
                "contradiction",
                topic_id=t["id"],
                support_type="stale",
                fact_group=fg_primary,
                is_current=False,
            )
        )
        doc_id += 1

        # 1 noise doc per topic
        documents.append(
            _build_doc(
                doc_id,
                t["noise"],
                f"news_{source_base}.txt",
                "news",
                "noise",
                topic_id=t["id"],
                support_type="noise",
                fact_group="noise",
            )
        )
        doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                _adversarial_summary(t),
                f"{source_base}_summary.txt",
                doc_type,
                "adversarial",
                topic_id=t["id"],
                support_type="adversarial",
                fact_group="summary",
                is_current=False,
            )
        )
        doc_id += 1

    # doc_id should be around 20 * 6 = 120 now; fill to 200 with extra noise
    noise_texts = [
        "TechNova Corp's board approved a $500 million share buyback program, citing strong free cash flow and robust balance sheet.",
        "The TechNova Corp mobile app reached 10 million downloads on the App Store, becoming the top-rated enterprise productivity tool.",
        "TechNova Corp expanded its partner ecosystem to 2,000 certified integration partners across 60 countries.",
        "TechNova Corp's security team blocked 2.3 billion malicious requests last month using its AI-powered threat detection system.",
        "TechNova Corp was named to the Fortune 100 Best Companies to Work For list for the third consecutive year.",
        "TechNova Corp's graduate hiring program accepted 350 candidates from 120 universities, focusing on AI, systems, and product roles.",
        "TechNova Corp reduced technical debt by 40% through a 6-month modernization initiative replacing legacy monolith services.",
        "TechNova Corp's EMEA revenue grew 32% year-over-year following the launch of dedicated EU data residency offerings.",
        "TechNova Corp achieved carbon neutrality for its data center operations through renewable energy procurement and efficiency improvements.",
        "TechNova Corp's developer platform now integrates with over 500 third-party tools including Jira, Slack, Datadog, and PagerDuty.",
        "TechNova Corp donated $2 million to STEM education initiatives in underserved communities across 15 cities.",
        "TechNova Corp's internal knowledge base was migrated to a vector search platform, reducing average support resolution time by 35%.",
        "TechNova Corp's API rate limits are 1,000 requests per minute for standard plans and 10,000 per minute for enterprise plans.",
        "TechNova Corp processes payments in 42 currencies using Stripe, with automatic tax calculation in 80 jurisdictions.",
        "TechNova Corp's test coverage reached 92% across all critical services following a dedicated quality improvement quarter.",
        "TechNova Corp's legal team successfully defended a patent portfolio of 340 patents in key markets including the US, EU, and Japan.",
        "TechNova Corp's product roadmap for 2025 includes native AI features, offline mode, and expanded enterprise admin controls.",
        "TechNova Corp's on-call rotation uses PagerDuty with a 5-minute escalation policy and 24/7 NOC support.",
        "TechNova Corp's data retention policy keeps customer data for 7 years in compliance with financial industry regulations.",
        "TechNova Corp's engineering ladders span L3 (junior) through L8 (distinguished engineer), with clear promotion criteria at each level.",
        "TechNova Corp's incident response team resolved 99.7% of P1 incidents within the 1-hour SLA in the past quarter.",
        "TechNova Corp's A/B testing platform runs over 200 simultaneous experiments with statistical significance tracking.",
        "TechNova Corp's localization team supports 28 languages and launched right-to-left (RTL) support for Arabic and Hebrew users.",
        "TechNova Corp's sales team closed deals with 250 new enterprise customers in Q3, representing $45 million in new ARR.",
        "TechNova Corp's feature flag system allows gradual rollouts to 0.1% of users, enabling safe deployment of major changes.",
        "TechNova Corp's compliance team maintains certifications for SOC 2, ISO 27001, GDPR, HIPAA, and FedRAMP Moderate.",
        "TechNova Corp's customer support team achieves a first-response time of under 2 hours for all paid plan customers.",
        "TechNova Corp's engineering blog published 48 technical articles last year, attracting 2 million unique readers.",
        "TechNova Corp's platform supports single sign-on via SAML 2.0 and OIDC, integrating with Okta, Azure AD, and Google Workspace.",
        "TechNova Corp's database layer uses PostgreSQL for transactional data and ClickHouse for analytics, with Redis for caching.",
        "TechNova Corp's mobile SDKs support iOS 15+, Android 8+, React Native, and Flutter.",
        "TechNova Corp's network infrastructure uses anycast routing to serve users from the nearest of 15 global PoPs.",
        "TechNova Corp's microservices architecture spans 340 services communicating via gRPC with an Envoy service mesh.",
        "TechNova Corp's observability stack includes Prometheus, Grafana, Jaeger distributed tracing, and OpenTelemetry.",
        "TechNova Corp's model training infrastructure supports up to 512 H100 GPUs in a single distributed training job.",
        "TechNova Corp's bug bounty program paid out $1.2 million to security researchers discovering 87 valid vulnerabilities.",
        "TechNova Corp's staging environment mirrors production at 1/10th scale and is refreshed with anonymized data weekly.",
        "TechNova Corp's deployment frequency averages 200 production deployments per day across all services.",
        "TechNova Corp's customer data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
        "TechNova Corp's machine learning models are retrained weekly using new user interaction data with automated quality gates.",
        "TechNova Corp's release process requires sign-off from two senior engineers, one QA lead, and one product manager.",
        "TechNova Corp's documentation portal receives 500,000 page views per month from developers integrating with its platform.",
        "TechNova Corp's analytics dashboard processes 10TB of event data daily using a Kafka-Flink-ClickHouse pipeline.",
        "TechNova Corp's SDK supports Python, JavaScript, Java, Go, Ruby, and C++ with idiomatic wrappers for each language.",
        "TechNova Corp's content delivery network caches static assets globally with a cache hit rate of 98.5%.",
        "TechNova Corp's identity system supports multi-factor authentication via TOTP, WebAuthn, and SMS as fallback.",
        "TechNova Corp's search functionality uses Elasticsearch with BM25 ranking augmented by a neural re-ranker.",
        "TechNova Corp's pricing model is usage-based, with costs calculated per API call, active user, and data storage tier.",
        "TechNova Corp's feature release notes are automatically generated from commit messages and Jira tickets using an LLM pipeline.",
        "TechNova Corp's on-premises deployment option uses Helm charts and supports air-gapped environments for government customers.",
        "TechNova Corp's data lake stores raw events in Parquet format on S3, partitioned by date and event type for cost-efficient queries.",
        "TechNova Corp's rollback procedure can revert any deployment to the previous version in under 90 seconds.",
        "TechNova Corp's GraphQL API was adopted by 40% of enterprise customers within 6 months of its launch.",
        "TechNova Corp's employee referral program awards $5,000 bonuses for referrals that result in successful hires.",
        "TechNova Corp's DevEx team maintains internal tooling including a CLI, VSCode extension, and Raycast plugin for developers.",
        "TechNova Corp's edge computing nodes process time-sensitive workloads within 5ms by running compute close to the user.",
        "TechNova Corp's chaos engineering practice uses Gremlin to inject faults weekly, improving system resilience scores by 15%.",
        "TechNova Corp's SRE team manages an error budget based on a 99.99% SLO, with burn rate alerts for rapid response.",
        "TechNova Corp's multi-tenant architecture uses row-level security in PostgreSQL to ensure strict data isolation between customers.",
        "TechNova Corp's webhook system delivers 500 million events per day with at-least-once delivery and idempotency keys.",
        "TechNova Corp's cost attribution system tags every cloud resource by team and product, enabling granular showback reporting.",
        "TechNova Corp's internal developer portal (IDP) was built on Backstage and catalogs all 340 microservices with runbooks.",
        "TechNova Corp's feature store serves pre-computed ML features with p99 latency under 5ms for online inference.",
        "TechNova Corp's automated regression suite runs 15,000 tests in under 10 minutes using parallelized test execution.",
        "TechNova Corp's API versioning policy maintains backward compatibility for 24 months after a version's deprecation notice.",
        "TechNova Corp's platform engineering team provisions new services using a self-service golden path with automated compliance checks.",
        "TechNova Corp's privacy team conducts annual data audits to ensure GDPR article 30 records of processing are up to date.",
        "TechNova Corp's on-call escalation policy requires P1 incidents to be acknowledged within 5 minutes or auto-escalated to the VP of Engineering.",
        "TechNova Corp's frontend performance budget mandates a Lighthouse score of 90+ for all customer-facing pages.",
        "TechNova Corp's internal AI assistant reduced time-to-answer for employee HR questions by 60% since its deployment.",
        "TechNova Corp's data warehouse uses dbt for transformation, with all models reviewed in pull requests before promotion.",
        "TechNova Corp's engineering principles include: build for reliability first, optimize for developer experience, and prefer boring technology.",
        "TechNova Corp's product analytics are collected via a first-party event pipeline to avoid third-party tracker dependencies.",
        "TechNova Corp's Kubernetes clusters run across 3 availability zones with node autoscaling and pod disruption budgets.",
        "TechNova Corp's error tracking system captures 10 million exceptions per day with automatic deduplication and priority ranking.",
        "TechNova Corp's platform passes annual pen tests conducted by an independent security firm with scope including API, web, and mobile.",
        "TechNova Corp's data masking pipeline anonymizes production snapshots before seeding into developer and QA environments.",
        "TechNova Corp's internal metrics show a 99.4% on-time delivery rate for product features committed to the quarterly roadmap.",
        "TechNova Corp's partner integration marketplace hosts 500 certified connectors, with 50 new connectors added each quarter.",
        "TechNova Corp's global support coverage ensures a 24/7 live engineer is available in every major timezone.",
        "TechNova Corp's engineering hiring funnel converted 2.1% of applicants to offers in 2024, with a median interview-to-offer time of 18 days.",
        "TechNova Corp's runtime error budgets are tracked weekly, and teams exceeding burn rate thresholds are required to freeze feature work.",
        "TechNova Corp's proprietary compression algorithm reduces log storage costs by 75% compared to standard gzip compression.",
        "TechNova Corp's API gateway handles certificate pinning for mobile clients, preventing man-in-the-middle attacks on enterprise deployments.",
        "TechNova Corp's hardware security module (HSM) stores all root encryption keys, ensuring cryptographic material never leaves secure hardware.",
        "TechNova Corp's regional failover drills are conducted quarterly to validate that RTO and RPO targets are achievable under realistic failure conditions.",
        "TechNova Corp's sales engineering team maintains live demo environments for each product tier, refreshed weekly with the latest stable build.",
    ]

    extra_needed = 200 - doc_id
    rng.shuffle(noise_texts)
    for text in noise_texts[:extra_needed]:
        documents.append(
            _build_doc(
                doc_id,
                text,
                f"technova_general_{doc_id}.txt",
                "news",
                "noise",
                topic_id=-1,
                support_type="noise",
            )
        )
        doc_id += 1

    queries = [_query_entry(t, phrasing="ops") for t in TOPIC_DEFINITIONS[:15]]

    return {"documents": documents, "queries": queries}


def _generate_hard_corpus(rng: random.Random) -> dict:
    """500 documents: all patterns including multi-hop."""
    documents = []
    doc_id = 0

    for t in TOPIC_DEFINITIONS:
        source_base = t["name"]
        doc_type = "technical" if t["is_multihop"] else "policy"
        fgs = _REQUIRED_FACT_GROUPS.get(t["name"], ["policy"])
        fg_primary = fgs[0]

        documents.append(
            _build_doc(
                doc_id,
                t["golden"],
                f"{source_base}_golden.txt",
                doc_type,
                "golden",
                topic_id=t["id"],
                support_type="partial" if t["is_multihop"] else "gold",
                fact_group="part1" if t["is_multihop"] else fg_primary,
            )
        )
        doc_id += 1

        for i, red in enumerate(t["redundants"]):
            documents.append(
                _build_doc(
                    doc_id,
                    red,
                    f"{source_base}_redundant_{i}.txt",
                    doc_type,
                    "redundant",
                    topic_id=t["id"],
                    support_type="partial" if t["is_multihop"] else "support",
                    fact_group="part1" if t["is_multihop"] else fg_primary,
                )
            )
            doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                t["contradiction"],
                f"{source_base}_contradiction.txt",
                doc_type,
                "contradiction",
                topic_id=t["id"],
                support_type="stale" if not t["is_multihop"] else "contradiction",
                fact_group="part1" if t["is_multihop"] else fg_primary,
                is_current=False,
            )
        )
        doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                t["noise"],
                f"news_{source_base}.txt",
                "news",
                "noise",
                topic_id=t["id"],
                support_type="noise",
            )
        )
        doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                _adversarial_summary(t),
                f"{source_base}_ambiguous.txt",
                doc_type,
                "adversarial",
                topic_id=t["id"],
                support_type="adversarial",
                fact_group="summary",
                is_current=False,
            )
        )
        doc_id += 1

        if t["is_multihop"] and t["part1"] and t["part2"]:
            documents.append(
                _build_doc(
                    doc_id,
                    t["part1"],
                    f"{source_base}_part1.txt",
                    "technical",
                    "multihop_part1",
                    topic_id=t["id"],
                    support_type="support",
                    fact_group="part1",
                )
            )
            doc_id += 1
            documents.append(
                _build_doc(
                    doc_id,
                    t["part2"],
                    f"{source_base}_part2.txt",
                    "technical",
                    "multihop_part2",
                    topic_id=t["id"],
                    support_type="support",
                    fact_group="part2",
                )
            )
            doc_id += 1

    # Fill remaining to 500 with extended noise bank
    extended_noise = [
        "TechNova Corp's AI platform uses a mixture-of-experts architecture with 8 specialized sub-models handling different task types.",
        "TechNova Corp's workflow automation tool processed 50 million tasks last month, saving an estimated 200,000 engineer-hours.",
        "TechNova Corp's multimodal AI supports text, image, audio, and structured data inputs for enterprise automation workflows.",
        "TechNova Corp's vector database stores 2 billion embeddings with sub-10ms p99 query latency using hierarchical navigable small worlds (HNSW).",
        "TechNova Corp's model fine-tuning service allows enterprises to adapt foundation models to domain-specific data in under 4 hours.",
        "TechNova Corp's synthetic data generation pipeline produces 10 million training examples per day for model improvement.",
        "TechNova Corp's streaming platform processes 2 trillion events per month using Apache Kafka with a 7-day retention window.",
        "TechNova Corp's real-time fraud detection model scores 500,000 transactions per second with 99.2% accuracy and 0.3% false positive rate.",
        "TechNova Corp's regulatory reporting module auto-generates 120 different report types for banking and insurance compliance.",
        "TechNova Corp's graph database stores knowledge relationships for 5 billion entities, enabling complex traversal queries.",
        "TechNova Corp's code generation tool has been adopted by 8,000 developers and reduces boilerplate writing time by 40%.",
        "TechNova Corp's federated learning framework allows model training across customer data without centralizing sensitive records.",
        "TechNova Corp's anomaly detection system identifies 98% of infrastructure incidents before they impact customer-facing services.",
        "TechNova Corp's time-series database compresses sensor data at 10:1 ratio while maintaining full query accuracy.",
        "TechNova Corp's natural language processing pipeline handles 28 languages with sub-200ms inference on CPU instances.",
        "TechNova Corp's data catalog automatically classifies 99% of new data assets within 10 minutes of ingestion.",
        "TechNova Corp's distributed cache uses a consistent hashing ring with 1024 virtual nodes for even load distribution.",
        "TechNova Corp's Terraform modules are used to provision infrastructure across 3 cloud providers from a single configuration.",
        "TechNova Corp's security posture management tool scans 500,000 cloud resources daily for misconfigurations.",
        "TechNova Corp's log aggregation system ingests 100TB of logs daily and retains searchable data for 90 days.",
        "TechNova Corp's model registry tracks 2,000 active ML model versions with lineage, metrics, and deployment status.",
        "TechNova Corp's platform health score aggregates 200 metrics into a single SRE reliability index updated every minute.",
        "TechNova Corp's contract intelligence tool extracts key terms from legal agreements with 97% accuracy vs. manual review.",
        "TechNova Corp's incident postmortem database contains 3,000 analyzed incidents, searchable by category, severity, and resolution type.",
        "TechNova Corp's unified observability platform ingests traces, metrics, and logs in OpenTelemetry format from all services.",
        "TechNova Corp's chaos mesh runs 50 fault injection experiments weekly across production-like staging environments.",
        "TechNova Corp's CI pipeline runs linting, type checking, unit tests, and integration tests in parallel within 4 minutes.",
        "TechNova Corp's ephemeral environments spin up a full stack for each pull request within 3 minutes using preview deployments.",
        "TechNova Corp's SLA dashboard gives customers real-time visibility into uptime, latency, and error rate against their commitments.",
        "TechNova Corp's cloud cost optimization resulted in $12 million annual savings through reserved instances and right-sizing.",
        "TechNova Corp's secrets management system rotates all service credentials automatically every 30 days without downtime.",
        "TechNova Corp's event-driven architecture uses a saga pattern for distributed transactions across 12 critical business flows.",
        "TechNova Corp's data pipeline SLA requires end-to-end latency of under 5 minutes for 99.9% of all data ingestion flows.",
        "TechNova Corp's platform API handles authentication via OAuth 2.0 with PKCE for public clients and client credentials for services.",
        "TechNova Corp's internal developer survey shows 94% of engineers rate the development environment as good or excellent.",
        "TechNova Corp's on-premises appliance version supports air-gapped deployment for government and defense sector customers.",
        "TechNova Corp's cross-region replication ensures RPO of 15 minutes and RTO of 30 minutes for disaster recovery scenarios.",
        "TechNova Corp's RBAC system supports 50 built-in roles and unlimited custom roles with attribute-based access control extensions.",
        "TechNova Corp's session management uses short-lived JWTs (15-minute expiry) with opaque refresh tokens stored in HttpOnly cookies.",
        "TechNova Corp's network egress costs were reduced 60% by implementing VPC endpoint routing for internal service communication.",
        "TechNova Corp's data anonymization service applies k-anonymity and differential privacy to protect individual user identities.",
        "TechNova Corp's container image scanning catches 100% of critical CVEs before images are promoted to production registries.",
        "TechNova Corp's model explainability tools generate SHAP values and LIME explanations for all production inference decisions.",
        "TechNova Corp's event sourcing system maintains a complete audit log of all domain events, enabling point-in-time recovery.",
        "TechNova Corp's developer experience metrics track time-to-first-PR (target: < 2 days) and deployment confidence scores.",
        "TechNova Corp's automated dependency update bot raises 150 PRs per week and auto-merges 70% when tests pass.",
        "TechNova Corp's internal platform team maintains 20 golden path templates covering web services, data pipelines, and ML workflows.",
        "TechNova Corp's service mesh collects per-RPC metrics with 1-second resolution for all 340 microservices.",
        "TechNova Corp's backup system performs point-in-time recovery tests monthly, achieving target RPO in 100% of simulations.",
        "TechNova Corp's semantic versioning policy requires major version bumps for any breaking API change.",
        "TechNova Corp's technical writing team produces API reference docs, tutorials, and conceptual guides for all public surfaces.",
        "TechNova Corp's platform processes credit card data in a PCI-DSS Level 1 compliant isolated environment.",
        "TechNova Corp's distributed lock service uses RedLock for cross-datacenter coordination with configurable TTLs.",
        "TechNova Corp's canary deployment system routes 1% of traffic to new versions before full rollout, with automatic rollback.",
        "TechNova Corp's feature parity matrix ensures all APIs are available in Python, Java, Go, and JavaScript SDKs simultaneously.",
        "TechNova Corp's event replay capability allows customers to re-process historical events up to 7 years in the past.",
        "TechNova Corp's model inference engine supports batching with dynamic batch sizes optimized for throughput vs. latency tradeoffs.",
        "TechNova Corp's data quality framework runs 5,000 assertion checks on every pipeline run, blocking promotion on failures.",
        "TechNova Corp's platform changelog is auto-generated from semantic release annotations and published to subscribers via webhook.",
        "TechNova Corp's identity verification service supports document scanning, liveness detection, and sanction list screening.",
        "TechNova Corp's multi-cloud strategy avoids vendor lock-in by abstracting cloud primitives behind an internal platform layer.",
        "TechNova Corp's release train deploys every Tuesday and Thursday, with hotfix capability for P0 incidents at any time.",
        "TechNova Corp's A/B testing framework uses Bayesian inference for faster decision-making, reducing experiment duration by 35%.",
        "TechNova Corp's platform engineering OKRs include: 99.999% control plane availability, < 5 min environment creation, 100% service in catalog.",
        "TechNova Corp's infrastructure is managed entirely as code, with Terraform plans reviewed in CI and applied via GitOps pipelines.",
        "TechNova Corp's global load balancer uses latency-based routing to direct users to the nearest healthy region.",
        "TechNova Corp's security champion program trains one engineer per team on OWASP Top 10 and secure code review practices.",
        "TechNova Corp's service catalog includes dependency graphs, SLO status, on-call contacts, and runbook links for all 340 services.",
        "TechNova Corp's platform health checks run every 10 seconds from 12 globally distributed probe locations.",
        "TechNova Corp's developer sandbox environment provides each engineer with a personal isolated namespace for safe experimentation.",
        "TechNova Corp's model governance process requires approval from ethics, legal, and product teams before production deployment.",
        "TechNova Corp's API gateway enforces rate limiting, request validation, authentication, and response transformation in a single pass.",
        "TechNova Corp's database connection pool is managed by PgBouncer, supporting 50,000 application connections via 200 server connections.",
        "TechNova Corp's artifact registry stores Docker images, Helm charts, npm packages, and Python wheels in a unified system.",
        "TechNova Corp's Slack integration delivers 2 million automated notifications per day covering alerts, deploys, and ticket updates.",
        "TechNova Corp's service reliability team runs quarterly Game Day exercises simulating regional failures and data corruption.",
        "TechNova Corp's latency budget allocates 50ms to the API gateway, 100ms to backend services, and 20ms to database queries.",
        "TechNova Corp's runtime security uses Falco to detect and alert on suspicious container behaviors in under 1 second.",
        "TechNova Corp's compliance automation checks 3,000 controls daily across cloud accounts, flagging drifts within 60 seconds.",
        "TechNova Corp's customer-visible status page is powered by Atlassian Statuspage and updated automatically from internal monitors.",
        "TechNova Corp's zero-downtime database migrations use shadow tables, dual-write patterns, and blue-green schema promotions.",
        "TechNova Corp's ML platform tracks carbon emissions per training run to help teams make energy-conscious model choices.",
        "TechNova Corp's internal AI assistant was built on a fine-tuned foundation model with company-specific tool use capabilities.",
        "TechNova Corp's customer data model supports multi-entity hierarchies up to 5 levels deep for complex enterprise org structures.",
        "TechNova Corp's developer portal integrates with 25 internal systems, providing a unified interface for provisioning, docs, and monitoring.",
        "TechNova Corp's privacy-preserving analytics uses aggregation with minimum cohort sizes of 50 to prevent re-identification.",
        "TechNova Corp's automated incident scoring model predicts severity within 30 seconds of the first alert, routing to the right team.",
        "TechNova Corp's platform supports webhook retry with exponential backoff, delivering events with 99.99% eventual success rate.",
        "TechNova Corp's distributed tracing captures 100% of production requests in staging and 10% sampled in production.",
        "TechNova Corp's internal CLI tool (tn) provides commands for deploying, rolling back, querying logs, and opening runbooks.",
        "TechNova Corp's API playground allows developers to test all endpoints interactively with live request/response inspection.",
        "TechNova Corp's SaaS application runs in a multi-tenant architecture with dedicated resources optionally available for enterprise tier.",
        "TechNova Corp's engineering onboarding program gets new hires to their first production deployment within 5 business days.",
        "TechNova Corp's data governance council reviews all new data collection proposals to ensure ethical use and regulatory compliance.",
        "TechNova Corp's platform billing system calculates usage in 5-minute windows and produces invoices with line-item granularity.",
        "TechNova Corp's mobile SDK automatically handles token refresh, retry logic, and offline queuing for unreliable network conditions.",
        "TechNova Corp's support escalation matrix routes critical issues to engineering on-call when resolution time exceeds 30 minutes.",
        "TechNova Corp's performance testing suite simulates 10x peak traffic weekly to identify bottlenecks before they affect customers.",
        "TechNova Corp's telemetry data is anonymized and aggregated at the edge before transmission to central analytics systems.",
        "TechNova Corp's documentation is version-controlled alongside code, requiring docs updates in the same PR as feature changes.",
        "TechNova Corp's multi-region database writes use a quorum of 3 replicas before acknowledging, ensuring strong consistency.",
        "TechNova Corp's AI-powered anomaly detection monitors 10,000 business metrics and pages on-call when patterns deviate significantly.",
        "TechNova Corp's identity federation supports just-in-time (JIT) provisioning, creating user accounts automatically on first SSO login.",
        "TechNova Corp's API deprecation policy gives customers 12 months notice before removing any endpoint or field.",
        "TechNova Corp's enterprise admin console supports bulk user management, audit log export, and custom data retention settings.",
        "TechNova Corp's machine learning platform supports experiment tracking with MLflow, serving with Triton, and pipelines with Kubeflow.",
        "TechNova Corp's platform API supports pagination via cursor-based tokens, ensuring stable results across large datasets.",
        "TechNova Corp's incident communication tool auto-drafts customer notifications using LLM summarization of internal incident chats.",
        "TechNova Corp's developer relations team manages 50 technology partnerships and publishes joint reference architectures quarterly.",
        "TechNova Corp's code review culture requires at least 2 approvals, with automated security review for changes in auth-related services.",
        "TechNova Corp's synthetic monitoring runs 500 user journey simulations every minute from 20 global locations.",
        "TechNova Corp's multi-factor enrollment rate reached 99.2% of active users following a mandatory MFA rollout for all plans.",
        "TechNova Corp's engineering metrics dashboard shows deployment frequency, lead time, MTTR, and change failure rate in real time.",
        "TechNova Corp's global expansion playbook guides market entry with localization, legal, and infrastructure checklists for 28 countries.",
        "TechNova Corp's API gateway plugin architecture allows customers to add custom middleware including auth, transformation, and logging.",
        "TechNova Corp's data pipeline monitoring checks source freshness every 5 minutes and alerts when data is more than 15 minutes stale.",
        "TechNova Corp's platform supports custom domains for customer-branded API endpoints with automated SSL certificate provisioning.",
        "TechNova Corp's internal tech radar classifies technologies into adopt, trial, assess, and hold categories, updated quarterly.",
        "TechNova Corp's service ownership model assigns each service an accountable team, on-call rotation, and quarterly reliability review.",
        "TechNova Corp's platform benchmarks show 99th percentile write latency of 8ms and read latency of 4ms for standard workloads.",
        "TechNova Corp's enterprise support tier includes dedicated Slack channels, named CSM, and 15-minute response SLA for P1 issues.",
        "TechNova Corp's data export API allows customers to retrieve all their data in JSON or CSV format within 24 hours of request.",
        "TechNova Corp's privacy portal gives end users visibility into what data is collected, with opt-out controls for non-essential processing.",
        "TechNova Corp's engineering handbook documents 200 architectural decision records (ADRs) explaining key technical choices made over time.",
        "TechNova Corp's platform gateway enforces payload size limits of 10MB for synchronous APIs and 1GB for async batch endpoints.",
        "TechNova Corp's infrastructure team manages 50,000 virtual machines and 200,000 container instances across 5 cloud regions.",
        "TechNova Corp's Kubernetes operators automate Day 2 operations including patching, scaling, certificate rotation, and backup verification.",
        "TechNova Corp's global content moderation team reviews flagged content across 28 languages with 4-hour target turnaround.",
        "TechNova Corp's internal security training is mandatory for all engineers annually and includes hands-on CTF challenges.",
        "TechNova Corp's revenue recognition system complies with ASC 606 and produces automated journal entries for each subscription event.",
    ]

    # ── Medical domain documents (5 topics × ~5 docs each = ~25 docs) ────────
    for t in MEDICAL_TOPICS:
        source_base = t["name"]
        fg = _REQUIRED_FACT_GROUPS.get(t["name"], ["medical"])[0]

        documents.append(
            _build_doc(
                doc_id,
                t["golden"],
                f"med_{source_base}_guideline.txt",
                "medical",
                "golden",
                topic_id=t["id"],
                support_type="gold",
                fact_group=fg,
            )
        )
        doc_id += 1

        for i, red in enumerate(t["redundants"][:2]):
            documents.append(
                _build_doc(
                    doc_id,
                    red,
                    f"med_{source_base}_ref_{i}.txt",
                    "medical",
                    "redundant",
                    topic_id=t["id"],
                    support_type="support",
                    fact_group=fg,
                )
            )
            doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                t["contradiction"],
                f"med_{source_base}_old.txt",
                "medical",
                "contradiction",
                topic_id=t["id"],
                support_type="stale",
                fact_group=fg,
                is_current=False,
            )
        )
        doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                t["noise"],
                f"med_{source_base}_news.txt",
                "news",
                "noise",
                topic_id=t["id"],
                support_type="noise",
                fact_group="noise",
            )
        )
        doc_id += 1

    # ── PyTorch / Meta / Scalar domain documents (5 topics × ~5 docs each) ──
    for t in PYTORCH_TOPICS:
        source_base = t["name"]
        doc_type = "technical" if not t["is_multihop"] else "technical"
        fgs = _REQUIRED_FACT_GROUPS.get(t["name"], ["pytorch"])
        fg_primary = fgs[0]

        documents.append(
            _build_doc(
                doc_id,
                t["golden"],
                f"pt_{source_base}_docs.txt",
                doc_type,
                "golden",
                topic_id=t["id"],
                support_type="partial" if t["is_multihop"] else "gold",
                fact_group="part1" if t["is_multihop"] else fg_primary,
            )
        )
        doc_id += 1

        for i, red in enumerate(t["redundants"][:2]):
            documents.append(
                _build_doc(
                    doc_id,
                    red,
                    f"pt_{source_base}_ref_{i}.txt",
                    doc_type,
                    "redundant",
                    topic_id=t["id"],
                    support_type="partial" if t["is_multihop"] else "support",
                    fact_group="part1" if t["is_multihop"] else fg_primary,
                )
            )
            doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                t["contradiction"],
                f"pt_{source_base}_old.txt",
                doc_type,
                "contradiction",
                topic_id=t["id"],
                support_type="stale",
                fact_group=fg_primary,
                is_current=False,
            )
        )
        doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                t["noise"],
                f"pt_{source_base}_news.txt",
                "news",
                "noise",
                topic_id=t["id"],
                support_type="noise",
                fact_group="noise",
            )
        )
        doc_id += 1

        documents.append(
            _build_doc(
                doc_id,
                _adversarial_summary(t),
                f"pt_{source_base}_ambiguous.txt",
                doc_type,
                "adversarial",
                topic_id=t["id"],
                support_type="adversarial",
                fact_group="summary",
                is_current=False,
            )
        )
        doc_id += 1

        if t["is_multihop"] and t["part1"] and t["part2"]:
            documents.append(
                _build_doc(
                    doc_id,
                    t["part1"],
                    f"pt_{source_base}_part1.txt",
                    doc_type,
                    "multihop_part1",
                    topic_id=t["id"],
                    support_type="support",
                    fact_group="part1",
                )
            )
            doc_id += 1
            documents.append(
                _build_doc(
                    doc_id,
                    t["part2"],
                    f"pt_{source_base}_part2.txt",
                    doc_type,
                    "multihop_part2",
                    topic_id=t["id"],
                    support_type="support",
                    fact_group="part2",
                )
            )
            doc_id += 1

    # ── Fill remaining to 500 with extended noise bank ─────────────────────
    extra_needed = 500 - doc_id
    rng.shuffle(extended_noise)
    for text in extended_noise[:extra_needed]:
        documents.append(
            _build_doc(
                doc_id,
                text,
                f"technova_extended_{doc_id}.txt",
                "news",
                "noise",
                topic_id=-1,
                support_type="noise",
            )
        )
        doc_id += 1

    # Pad if still short
    while len(documents) < 500:
        documents.append(
            _build_doc(
                doc_id,
                f"TechNova Corp technical document {doc_id}: internal reference material for engineering and operations teams.",
                f"internal_ref_{doc_id}.txt",
                "technical",
                "noise",
                topic_id=-1,
                support_type="noise",
            )
        )
        doc_id += 1

    # 10 TechNova + 5 medical + 5 PyTorch/Meta/Scalar = 20 queries (n_queries=20 for hard)
    tn_queries = [_query_entry(t, phrasing="audit") for t in TOPIC_DEFINITIONS[:10]]
    med_queries = [_query_entry(t) for t in MEDICAL_TOPICS]
    pt_queries = [_query_entry(t) for t in PYTORCH_TOPICS]
    queries = (tn_queries + med_queries + pt_queries)[:20]

    return {"documents": documents, "queries": queries}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_corpus(task: TaskConfig) -> dict[str, Any]:
    rng = random.Random(42)
    if task.name == "easy":
        return _generate_easy_corpus(rng)
    elif task.name == "medium":
        return _generate_medium_corpus(rng)
    elif task.name == "hard":
        return _generate_hard_corpus(rng)
    raise ValueError(f"Unknown task: {task.name}")


def load_or_generate_corpus(task: TaskConfig, corpus_dir: str = "corpora") -> dict[str, Any]:
    path = os.path.join(corpus_dir, f"{task.name}.json")
    if os.path.exists(path):
        with open(path) as f:
            corpus = json.load(f)
        documents = corpus.get("documents", [])
        queries = corpus.get("queries", [])
        has_metadata = bool(documents) and all(
            "support_type" in d and "topic_id" in d and "fact_group" in d for d in documents[:10]
        )
        if has_metadata and len(queries) == task.n_queries:
            return corpus
    corpus = generate_corpus(task)
    os.makedirs(corpus_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(corpus, f, indent=2)
    return corpus
