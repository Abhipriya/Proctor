from fastapi import FastAPI, BackgroundTasks, HTTPException
from datetime import datetime, timedelta
import threading
import time

app = FastAPI()

sessions = {}

SESSION_DURATION = 30  # 30 seconds for testing (change to 5400 for 90 mins)


def auto_submit_session(session_id):
    print(f"[AutoSubmit] Auto submission started for session: {session_id}")
    time.sleep(SESSION_DURATION)  # Wait for the session duration to pass
    session = sessions.get(session_id)

    if session:
        if not session.get("submitted", False):
            print(f"[AutoSubmit] Submitting session {session_id} after timeout.")
            session["submitted"] = True
            session["active"] = False
        else:
            print(f"[AutoSubmit] Session {session_id} already submitted.")
    else:
        print(f"[AutoSubmit] Session {session_id} not found.")


@app.post("/start-session/{session_id}")
def start_session(session_id: str, background_tasks: BackgroundTasks):
    sessions[session_id] = {
        "start_time": datetime.now(),
        "warnings": 0,
        "submitted": False,
        "active": True
    }
    print(f"[Session] Started session {session_id}")
    background_tasks.add_task(auto_submit_session, session_id)
    return {"message": f"Session {session_id} started."}


@app.post("/warning/{session_id}")
def add_warning(session_id: str):
    session = sessions.get(session_id)
    if not session or not session.get("active", False):
        raise HTTPException(status_code=404, detail="Session not found or already ended.")

    session["warnings"] += 1
    print(f"[Warning] Session {session_id} now has {session['warnings']} warnings.")
    if session["warnings"] >= 3:
        session["active"] = False
        session["submitted"] = True
        print(f"[Warning] Session {session_id} terminated after 3 warnings.")
        return {"message": "Session terminated after 3 warnings."}

    return {"message": f"Warning added. Current count: {session['warnings']}"}


@app.get("/session-status/{session_id}")
def get_session_status(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session




