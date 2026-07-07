#A comprehensive Streamlit application for the Authorization & Entitlement Intelligence Agent using Generative Engine's API.
# This implementation includes a complete entitlement system with AI-powered decision-making.

import streamlit as st
import json
import uuid
from datetime import datetime
from openai import OpenAI
import os

# Page config
st.set_page_config(
    page_title="Authorization & Entitlement Intelligence Agent",
    page_icon="🔐",
    layout="wide"
)

# Initialize session state
if 'decision_history' not in st.session_state:
    st.session_state.decision_history = []

# Get API key from environment or text input
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

# Sample entitlement database
ENTITLEMENT_DATABASE = {
    "users": {
        "john.doe": {
            "role": "Account Manager",
            "level": "Senior",
            "entitlements": ["view_account", "modify_account", "approve_transactions_under_50k"],
            "accounts": ["ACC001", "ACC002", "ACC003"],
            "approval_limit": 50000,
            "department": "Operations"
        },
        "jane.smith": {
            "role": "Team Lead",
            "level": "Lead",
            "entitlements": ["view_account", "modify_account", "approve_transactions_under_100k", "manage_team"],
            "accounts": ["ACC001", "ACC002", "ACC003", "ACC004", "ACC005"],
            "approval_limit": 100000,
            "department": "Operations"
        },
        "mike.johnson": {
            "role": "Junior Analyst",
            "level": "Junior",
            "entitlements": ["view_account", "create_report"],
            "accounts": ["ACC001", "ACC002"],
            "approval_limit": 0,
            "department": "Analytics"
        },
        "sarah.williams": {
            "role": "VP Operations",
            "level": "Executive",
            "entitlements": ["view_account", "modify_account", "approve_transactions_unlimited", "manage_team", "override_rules"],
            "accounts": ["ALL"],
            "approval_limit": 999999,
            "department": "Executive"
        },
        "tom.brown": {
            "role": "Compliance Officer",
            "level": "Senior",
            "entitlements": ["view_account", "audit_trail", "approve_high_risk"],
            "accounts": ["ALL"],
            "approval_limit": 75000,
            "department": "Compliance"
        }
    },
    "transaction_rules": {
        "transfer": {
            "min_level": "Senior",
            "requires_approval": True,
            "approval_threshold": 10000,
            "restricted_accounts": ["ACC005"]
        },
        "withdrawal": {
            "min_level": "Senior",
            "requires_approval": True,
            "approval_threshold": 5000,
            "restricted_accounts": ["ACC005"]
        },
        "deposit": {
            "min_level": "Junior",
            "requires_approval": False,
            "approval_threshold": 999999,
            "restricted_accounts": []
        },
        "account_modification": {
            "min_level": "Senior",
            "requires_approval": True,
            "approval_threshold": 0,
            "restricted_accounts": ["ACC004", "ACC005"]
        },
        "view_statement": {
            "min_level": "Junior",
            "requires_approval": False,
            "approval_threshold": 0,
            "restricted_accounts": []
        },
        "close_account": {
            "min_level": "Lead",
            "requires_approval": True,
            "approval_threshold": 0,
            "restricted_accounts": ["ACC005"]
        }
    },
    "account_relationships": {
        "ACC001": {"type": "Corporate", "status": "Active", "risk_level": "Low", "balance": 500000},
        "ACC002": {"type": "Individual", "status": "Active", "risk_level": "Medium", "balance": 150000},
        "ACC003": {"type": "Corporate", "status": "Active", "risk_level": "High", "balance": 2000000},
        "ACC004": {"type": "Corporate", "status": "Under Review", "risk_level": "High", "balance": 750000},
        "ACC005": {"type": "Special", "status": "Restricted", "risk_level": "Critical", "balance": 5000000}
    }
}

# Pre-defined test scenarios
TEST_SCENARIOS = {
    "Select a scenario...": None,
    "✅ Success: Senior Manager Small Transfer": {
        "username": "john.doe",
        "transaction_type": "transfer",
        "account_id": "ACC001",
        "amount": 25000,
        "context": "Standard business transfer within approval limits"
    },
    "❌ Denied: Junior High-Value Transfer": {
        "username": "mike.johnson",
        "transaction_type": "transfer",
        "account_id": "ACC002",
        "amount": 75000,
        "context": "Junior analyst attempting high-value transfer exceeding authorization"
    },
    "⚠️ Approval Needed: Restricted Account Access": {
        "username": "jane.smith",
        "transaction_type": "withdrawal",
        "account_id": "ACC005",
        "amount": 30000,
        "context": "Team lead attempting withdrawal from restricted/special account"
    },
    "✅ Success: Executive Override": {
        "username": "sarah.williams",
        "transaction_type": "transfer",
        "account_id": "ACC005",
        "amount": 150000,
        "context": "VP Operations with executive privileges on special account"
    },
    "⚠️ Approval Needed: Account Under Review": {
        "username": "john.doe",
        "transaction_type": "account_modification",
        "account_id": "ACC004",
        "amount": 0,
        "context": "Modification attempt on account currently under compliance review"
    },
    "❌ Denied: Unauthorized Account Access": {
        "username": "mike.johnson",
        "transaction_type": "transfer",
        "account_id": "ACC003",
        "amount": 5000,
        "context": "Junior analyst accessing account outside their authorized list"
    },
    "✅ Success: Simple View Request": {
        "username": "mike.johnson",
        "transaction_type": "view_statement",
        "account_id": "ACC001",
        "amount": 0,
        "context": "Junior analyst viewing statement of authorized account"
    },
    "⚠️ Approval Needed: High-Risk Transaction": {
        "username": "john.doe",
        "transaction_type": "transfer",
        "account_id": "ACC003",
        "amount": 55000,
        "context": "Transfer exceeding approval limit on high-risk account"
    }
}

def check_authorization_with_ai(username, transaction_type, account_id, amount, additional_context, api_key):
    """Check authorization using AI agent"""
    
    try:
        user_context = ENTITLEMENT_DATABASE["users"].get(username.lower(), None)
        transaction_rules = ENTITLEMENT_DATABASE["transaction_rules"].get(transaction_type.lower(), None)
        account_info = ENTITLEMENT_DATABASE["account_relationships"].get(account_id, None)
        
        # Find potential approvers
        potential_approvers = []
        for uname, udata in ENTITLEMENT_DATABASE["users"].items():
            if uname.lower() != username.lower():
                if amount > 0 and udata["approval_limit"] >= amount:
                    potential_approvers.append(f"{uname} ({udata['role']}, Limit: ${udata['approval_limit']:,})")
                elif udata["level"] in ["Lead", "Executive"]:
                    potential_approvers.append(f"{uname} ({udata['role']}, {udata['level']})")
        
        # Build comprehensive prompt
        prompt = f"""You are an Authorization & Entitlement Intelligence Agent for a financial services platform.

AUTHORIZATION REQUEST:
User: {username}
Transaction Type: {transaction_type}
Account: {account_id}
Amount: ${amount:,} USD
Context: {additional_context}

USER PROFILE:
{json.dumps(user_context, indent=2) if user_context else "⚠️ USER NOT FOUND IN SYSTEM"}

TRANSACTION RULES FOR '{transaction_type}':
{json.dumps(transaction_rules, indent=2) if transaction_rules else "⚠️ NO RULES DEFINED"}

ACCOUNT INFORMATION FOR '{account_id}':
{json.dumps(account_info, indent=2) if account_info else "⚠️ ACCOUNT NOT FOUND"}

POTENTIAL APPROVERS IN SYSTEM:
{chr(10).join(potential_approvers) if potential_approvers else "No approvers available"}

YOUR TASK:
Analyze this authorization request and determine:
1. Should this be AUTHORIZED, NOT AUTHORIZED, or REQUIRES APPROVAL?
2. What are the specific reasons for this decision?
3. What factors did you consider?
4. If denied, what are the blocking factors?
5. If approval needed, who can approve and what steps are required?
6. Are there alternate approaches the user can take?
7. What is the risk assessment?

CRITICAL CHECKS:
- Does user have required entitlements?
- Is user authorized for this account? (check if account is in user's account list or if user has "ALL" access)
- Does transaction amount exceed user's approval limit?
- Is account status problematic? (Under Review, Restricted)
- Is this a restricted account for this transaction type?
- Does user's level meet minimum requirements?
- Is account risk level compatible with user's authorization level?

Respond in VALID JSON format only:
{{
    "decision": "AUTHORIZED" | "NOT AUTHORIZED" | "REQUIRES APPROVAL",
    "confidence": "HIGH" | "MEDIUM" | "LOW",
    "explanation": "Clear, detailed explanation in 2-3 sentences",
    "factors_considered": ["factor1", "factor2", "factor3"],
    "blocking_factors": ["reason1", "reason2"] or [],
    "required_next_steps": ["step1", "step2"] or [],
    "alternate_paths": ["path1", "path2"] or [],
    "recommended_approvers": ["approver1", "approver2"] or [],
    "compliance_notes": "Any compliance concerns",
    "risk_assessment": "Risk level and specific concerns"
}}"""

        # Initialize OpenAI client
        client = OpenAI(
            base_url="https://openai.generative.engine.capgemini.com/v1",
            api_key=api_key
        )
        
        # Call API
        response = client.chat.completions.create(
            model="amazon.nova-pro-v1:0",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial authorization system. Provide accurate, explainable decisions in valid JSON format. Be strict about security but helpful with guidance."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        # Parse response
        ai_response = response.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in ai_response:
            ai_response = ai_response.split("```json")[1].split("```")[0].strip()
        elif "```" in ai_response:
            ai_response = ai_response.split("```")[1].split("```")[0].strip()
        
        decision_data = json.loads(ai_response)
        
        # Add metadata
        decision_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        decision_data["request_id"] = str(uuid.uuid4())[:8]
        decision_data["context"] = {
            "user": user_context,
            "rules": transaction_rules,
            "account": account_info
        }
        
        return decision_data, None
        
    except json.JSONDecodeError as e:
        return None, f"Failed to parse AI response as JSON: {str(e)}\n\nRaw response:\n{ai_response[:500]}"
    except Exception as e:
        return None, f"Error: {str(e)}"

def display_user_info(username):
    """Display user information card"""
    user = ENTITLEMENT_DATABASE["users"].get(username.lower())
    if user:
        st.info(f"""
        **👤 {username}**  
        **Role:** {user['role']} ({user['level']})  
        **Department:** {user['department']}  
        **Approval Limit:** ${user['approval_limit']:,}  
        **Entitlements:** {', '.join(user['entitlements'][:3])}{'...' if len(user['entitlements']) > 3 else ''}  
        **Authorized Accounts:** {', '.join(user['accounts']) if user['accounts'][0] != 'ALL' else 'ALL ACCOUNTS'}
        """)

def display_account_info(account_id):
    """Display account information card"""
    account = ENTITLEMENT_DATABASE["account_relationships"].get(account_id)
    if account:
        status_emoji = "🟢" if account["status"] == "Active" else "🟡" if account["status"] == "Under Review" else "🔴"
        risk_emoji = "🟢" if account["risk_level"] == "Low" else "🟡" if account["risk_level"] == "Medium" else "🟠" if account["risk_level"] == "High" else "🔴"
        
        st.info(f"""
        **💳 {account_id}**  
        **Type:** {account['type']}  
        **Status:** {status_emoji} {account['status']}  
        **Risk Level:** {risk_emoji} {account['risk_level']}  
        **Balance:** ${account['balance']:,}
        """)

def display_decision_result(decision):
    """Display authorization decision"""
    
    st.markdown("---")
    
    # Decision header with large badge
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if decision["decision"] == "AUTHORIZED":
            st.success(f"# ✅ {decision['decision']}")
        elif decision["decision"] == "REQUIRES APPROVAL":
            st.warning(f"# ⚠️ {decision['decision']}")
        else:
            st.error(f"# ❌ {decision['decision']}")
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Confidence", decision.get("confidence", "N/A"))
    with col2:
        st.metric("Request ID", decision.get("request_id", "N/A"))
    with col3:
        st.metric("Timestamp", decision.get("timestamp", "N/A"))
    
    # Main explanation
    st.markdown("### 📝 Decision Explanation")
    st.info(decision.get("explanation", "No explanation provided"))
    
    # Two column layout for details
    col1, col2 = st.columns(2)
    
    with col1:
        # Factors considered
        if decision.get("factors_considered"):
            st.markdown("### 🔍 Factors Considered")
            for factor in decision["factors_considered"]:
                st.write(f"✓ {factor}")
        
        # Blocking factors
        if decision.get("blocking_factors"):
            st.markdown("### 🚫 Blocking Factors")
            for factor in decision["blocking_factors"]:
                st.error(f"✗ {factor}")
    
    with col2:
        # Required next steps
        if decision.get("required_next_steps"):
            st.markdown("### 📌 Required Next Steps")