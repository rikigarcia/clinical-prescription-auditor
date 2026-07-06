# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any

import google.auth
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.workflow import Workflow, node
from google.genai import types
from pydantic import BaseModel, Field

# Setup authentication dynamically
try:
    _, project_id = google.auth.default()
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    pass

os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


class PrescriptionDetails(BaseModel):
    patient_name: str = Field(description="The name of the patient.")
    patient_age: int = Field(description="The age of the patient in years.")
    patient_allergies: list[str] = Field(
        default=[], description="List of patient drug allergies."
    )
    prescribed_drug: str = Field(
        description="The name of the drug being prescribed (e.g., Amoxicillin, Ibuprofen)."
    )
    dosage: str = Field(
        description="The dosage instructions (e.g., 500mg daily, twice a day)."
    )
    symptoms: str = Field(
        default="", description="The symptoms or medical condition being treated."
    )


class ClinicalAuditReport(BaseModel):
    is_safe: bool = Field(
        description="Whether the prescription is considered safe for the patient."
    )
    risk_score: float = Field(
        description="A safety risk score between 0.0 (no risk) and 1.0 (fatal risk)."
    )
    conflicts_found: list[str] = Field(
        default=[], description="Specific conflicts (e.g. allergy cross-reactions)."
    )
    warnings: list[str] = Field(
        default=[], description="General drug warnings or age-related precautions."
    )
    clinical_explanation: str = Field(
        description="Explanation of the safety and risk assessment."
    )


def check_security(node_input: Any) -> Event:
    """Performs pre-execution security checks, redacting PII and catching prompt injections."""
    text = ""
    if isinstance(node_input, types.Content):
        text = "".join(part.text for part in node_input.parts if part.text)
    elif isinstance(node_input, dict):
        text = (
            str(node_input.get("symptoms", ""))
            + " "
            + str(node_input.get("patient_name", ""))
        )
    else:
        text = str(node_input)

    text_lower = text.lower()

    # 1. Catch prompt injection / override attempts
    injection_triggers = [
        "ignore previous",
        "ignore the above",
        "system override",
        "override safety",
        "you must approve",
        "bypass clinical",
        "ignore all instructions",
        "bypass audit",
        "override compliance",
    ]

    is_flagged = any(trigger in text_lower for trigger in injection_triggers)
    if is_flagged:
        return Event(
            output={
                "status": "REJECTED",
                "reason": "Security Alert: Malicious instructions or prompt injection detected.",
            },
            route="reject_security",
            state={"security_flag": True, "raw_input": text},
        )

    # 2. Basic PII Redaction (e.g., Redacting standard Social Security Numbers if accidentally input)
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    phone_pattern = r"\b\d{3}-\d{3}-\d{4}\b"

    redacted_text = re.sub(ssn_pattern, "[REDACTED SSN]", text)
    redacted_text = re.sub(phone_pattern, "[REDACTED PHONE]", redacted_text)

    cleaned_input = node_input
    if isinstance(cleaned_input, types.Content):
        for part in cleaned_input.parts:
            if part.text:
                part.text = re.sub(ssn_pattern, "[REDACTED SSN]", part.text)
                part.text = re.sub(phone_pattern, "[REDACTED PHONE]", part.text)

    return Event(output=cleaned_input, route="proceed")


def search_fda_drug_label(brand_name: str) -> dict:
    """Queries openFDA API for drug labeling information, warnings, and contraindications.

    Args:
        brand_name: The brand name or generic name of the drug.

    Returns:
        A dictionary containing warnings, contraindications, and usage indications.
    """
    name = brand_name.strip()
    query = f'openfda.brand_name.exact:"{name.upper()}"+OR+openfda.generic_name.exact:"{name.lower()}"'
    url = f"https://api.fda.gov/drug/label.json?search={urllib.parse.quote(query)}&limit=1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                return {
                    "brand_name": brand_name,
                    "warnings": result.get("warnings", ["Standard precautions apply."])[
                        0
                    ][:500],
                    "contraindications": result.get(
                        "contraindications", ["No specific contraindications listed."]
                    )[0][:500],
                    "indications_and_usage": result.get(
                        "indications_and_usage", ["Used as medically indicated."]
                    )[0][:500],
                }
    except Exception:
        pass

    # Return stable default warnings if API is rate-limited or offline
    return {
        "brand_name": brand_name,
        "warnings": f"Standard precautions apply for {brand_name}. Monitor patient for side effects.",
        "contraindications": f"Known hypersensitivity to {brand_name} or related compounds.",
        "indications_and_usage": f"Indicated for treatment according to standard clinical protocols for {brand_name}.",
    }


def check_drug_allergy_contraindications(
    prescribed_drug: str, patient_allergies: list[str]
) -> dict:
    """Checks for known clinical cross-reactivity or allergy conflicts.

    Args:
        prescribed_drug: The name of the drug being prescribed.
        patient_allergies: List of patient allergies.

    Returns:
        A dictionary indicating conflict status and specific reasons.
    """
    drug_lower = prescribed_drug.lower().strip()
    allergies_lower = [a.lower().strip() for a in patient_allergies]

    conflicts = []

    # Common cross-reactivity mapping
    penicillins = [
        "amoxicillin",
        "ampicillin",
        "penicillin",
        "piperacillin",
    ]
    nsaids = ["ibuprofen", "aspirin", "naproxen", "diclofenac", "celecoxib"]
    sulfas = ["sulfamethoxazole", "bactrim", "dapsone"]

    for allergy in allergies_lower:
        if "penicillin" in allergy:
            if any(p in drug_lower for p in penicillins):
                conflicts.append(
                    f"Patient penicillin allergy cross-reacts with prescribed beta-lactam ({prescribed_drug})."
                )
        if "nsaid" in allergy or "aspirin" in allergy or "ibuprofen" in allergy:
            if any(n in drug_lower for n in nsaids):
                conflicts.append(
                    f"Patient NSAID/Aspirin allergy cross-reacts with prescribed NSAID ({prescribed_drug})."
                )
        if "sulfa" in allergy:
            if any(s in drug_lower for s in sulfas):
                conflicts.append(
                    f"Patient sulfa allergy cross-reacts with prescribed sulfonamide ({prescribed_drug})."
                )
        if allergy in drug_lower or drug_lower in allergy:
            conflicts.append(
                f"Direct match: Patient allergic to {allergy}; prescribed drug is {prescribed_drug}."
            )

    if conflicts:
        return {"status": "CONFLICT", "conflicts": conflicts}
    return {"status": "SAFE", "conflicts": []}


# LLM-based prescription details extraction agent
extraction_agent = LlmAgent(
    name="extraction_agent",
    model=Gemini(model="gemini-flash-latest"),
    instruction="""
    You are an expert medical transcriptionist. Extract the prescription details from the doctor's unstructured input text.
    Make sure to capture:
    - patient_name
    - patient_age (integer)
    - patient_allergies (list of strings, leave empty if none mentioned)
    - prescribed_drug
    - dosage
    - symptoms

    If any detail like symptoms is not mentioned, leave it empty.
    If the text does not contain a prescription request, set prescribed_drug to 'None' and dosage to 'None'.
    """,
    output_schema=PrescriptionDetails,
    output_key="prescription",
)

# LLM-based clinical auditor agent
clinical_auditor = LlmAgent(
    name="clinical_auditor",
    model=Gemini(model="gemini-flash-latest"),
    instruction="""
    You are a clinical safety auditor. Audit the patient's prescription against the clinical context, FDA label search results, and allergy checks.

    Standard Rules:
    1. Severe Allergy conflicts: If any conflicts are flagged in the allergy check, set is_safe=False, risk_score >= 0.9, and note the cross-reactivity.
    2. Pediatric/Geriatric warnings:
       - Pediatric (age < 12): Flag warnings if the prescribed drug contains warnings regarding children or if the FDA label warnings mention child precautions.
       - Geriatric (age > 65): Flag warnings if the FDA warnings or contraindications mention elderly precautions.
    3. If the prescribed drug is 'None', set is_safe=False, risk_score=1.0, and add 'Invalid prescription details' to conflicts_found.

    Search reference state context:
    - FDA warnings: {fda_warnings}
    - FDA contraindications: {fda_contraindications}
    - Allergy conflicts: {allergy_conflicts}

    Output a structured ClinicalAuditReport with:
    - is_safe (boolean)
    - risk_score (float, 0.0 to 1.0)
    - conflicts_found (list of strings)
    - warnings (list of strings)
    - clinical_explanation (brief summary of findings)
    """,
    output_schema=ClinicalAuditReport,
    output_key="audit_report",
)


def evaluate_audit(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Evaluates the audit report and routes to approval, rejection, or physician review."""
    is_safe = node_input.get("is_safe", True)
    risk_score = float(node_input.get("risk_score", 0.0))
    conflicts = node_input.get("conflicts_found", [])

    prescription = ctx.state.get("prescription", {})
    drug = prescription.get("prescribed_drug", "None")

    if drug == "None" or "Invalid prescription details" in conflicts:
        return Event(
            output={
                "status": "REJECTED",
                "reason": "Invalid or incomplete prescription request.",
            },
            route="auto_reject",
        )

    # Auto-approve only if safe, risk score is low (< 0.25), and there are no severe conflicts
    if is_safe and risk_score < 0.25 and len(conflicts) == 0:
        return Event(output=prescription, route="auto_approve")
    elif risk_score >= 0.85:
        # Severe safety risks are auto-rejected
        return Event(
            output={
                "status": "REJECTED",
                "reason": f"Safety audit failed: {', '.join(conflicts)}",
            },
            route="auto_reject",
        )
    else:
        # Borderline risk or general precautions route to Physician/Doctor-in-the-loop review
        return Event(output=prescription, route="review_prescription")


def auto_approve(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Instantly approves safe prescriptions."""
    prescription = ctx.state.get("prescription", node_input)
    patient = prescription.get("patient_name", "Patient")
    drug = prescription.get("prescribed_drug", "Drug")
    dosage = prescription.get("dosage", "dosage info")

    audit_report = ctx.state.get("audit_report", {})
    explanation = audit_report.get("clinical_explanation", "Passed safety checks.")

    outcome = {
        "status": "APPROVED",
        "reason": f"Auto-approved: {explanation}",
        "patient": patient,
        "drug": drug,
    }
    result_text = (
        f"Prescription APPROVED: {drug} ({dosage}) for {patient} has been automatically approved.\n"
        f"Clinical check: {explanation}"
    )
    return Event(
        output=outcome,
        state={"outcome": outcome},
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=result_text)]
        ),
    )


def auto_reject(node_input: dict[str, Any]) -> Event:
    """Handles automatic rejections for high-risk prescriptions."""
    outcome = {
        "status": "REJECTED",
        "reason": node_input.get(
            "reason", "Contraindicated or unsafe prescription request."
        ),
        "patient": "system",
        "drug": "None",
    }
    result_text = f"Prescription REJECTED: {outcome['reason']}"
    return Event(
        output=outcome,
        state={"outcome": outcome},
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=result_text)]
        ),
    )


def reject_security(node_input: dict[str, Any]) -> Event:
    """Handles security blocks (e.g. prompt injection attempts)."""
    outcome = {
        "status": "REJECTED",
        "reason": node_input.get("reason", "PII validation or injection check failed."),
        "patient": "system",
        "drug": "None",
    }
    result_text = f"Security Block: {outcome['reason']}"
    return Event(
        output=outcome,
        state={"outcome": outcome},
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=result_text)]
        ),
    )


@node(rerun_on_resume=True)
async def review_prescription(ctx: Context, node_input: dict[str, Any]):
    """Physician/Pharmacist human-in-the-loop review node."""
    prescription = ctx.state.get("prescription", node_input)
    patient = prescription.get("patient_name", "Patient")
    drug = prescription.get("prescribed_drug", "Drug")
    dosage = prescription.get("dosage", "dosage info")

    audit_report = ctx.state.get("audit_report", {})
    reasons = (
        ", ".join(
            audit_report.get("conflicts_found", []) + audit_report.get("warnings", [])
        )
        or "Precautionary warning"
    )
    explanation = audit_report.get(
        "clinical_explanation", "Requires physician confirmation."
    )

    message = (
        f"PHYSICIAN AUDIT REVIEW REQUIRED ({patient} - {drug} {dosage}).\n"
        f"Reason(s) flagged: {reasons}\n"
        f"Clinical report: {explanation}\n"
        f"Please reply with 'approve' or 'reject' to finalize:"
    )

    interrupt_id = "physician_decision"

    if not ctx.resume_inputs or interrupt_id not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id=interrupt_id,
            message=message,
        )
        return

    decision_val = ctx.resume_inputs[interrupt_id]
    if isinstance(decision_val, str):
        try:
            decision_val = json.loads(decision_val)
        except Exception:
            pass

    is_approved = False
    if isinstance(decision_val, dict):
        if "approved" in decision_val:
            is_approved = bool(decision_val["approved"])
        elif "result" in decision_val:
            is_approved = "approve" in str(decision_val["result"]).lower()
    else:
        decision_str = str(decision_val).strip().lower()
        is_approved = "approve" in decision_str and "reject" not in decision_str

    status = "APPROVED" if is_approved else "REJECTED"

    outcome = {
        "status": status,
        "reason": f"Manual override by physician: {decision_val}",
        "patient": patient,
        "drug": drug,
    }
    result_text = f"Prescription of {drug} for {patient} was manually {status}."

    yield Event(
        output=outcome,
        state={"outcome": outcome},
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=result_text)]
        ),
    )


# Node function to fetch FDA info and run allergy check to populate clinical_auditor context
def query_clinical_databases(ctx: Context, node_input: PrescriptionDetails) -> Event:
    """Pre-computes drug warning and allergy information to guide auditor reasoning."""
    drug = node_input.prescribed_drug
    allergies = node_input.patient_allergies

    fda_info = search_fda_drug_label(drug)
    allergy_info = check_drug_allergy_contraindications(drug, allergies)

    # Store findings in state delta for the auditor agent's dynamic system prompt injection
    state_delta = {
        "fda_warnings": fda_info.get("warnings", "No warnings listed."),
        "fda_contraindications": fda_info.get(
            "contraindications", "No specific contraindications."
        ),
        "allergy_conflicts": ", ".join(allergy_info.get("conflicts", []))
        or "None detected.",
    }

    return Event(output=node_input, state=state_delta)


# Define workflow routing graph
root_agent = Workflow(
    name="prescription_auditor_workflow",
    edges=[
        ("START", check_security),
        (
            check_security,
            {
                "proceed": extraction_agent,
                "reject_security": reject_security,
            },
        ),
        (extraction_agent, query_clinical_databases),
        (query_clinical_databases, clinical_auditor),
        (clinical_auditor, evaluate_audit),
        (
            evaluate_audit,
            {
                "auto_approve": auto_approve,
                "auto_reject": auto_reject,
                "review_prescription": review_prescription,
            },
        ),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
