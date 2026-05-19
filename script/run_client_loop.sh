#!/usr/bin/env bash
set -u

CMD=(python client.py -p ~/The_Economist_0703.pdf --timeout 3600)
INTERVAL_SEC="${INTERVAL_SEC:-0}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"
LOG_DIR="${LOG_DIR:-./loop_logs}"

mkdir -p "$LOG_DIR"
summary_file="$LOG_DIR/summary_$(date +%Y%m%d_%H%M%S).log"

echo "Start loop at $(date '+%F %T')" | tee -a "$summary_file"
echo "Command: ${CMD[*]}" | tee -a "$summary_file"
echo "INTERVAL_SEC=$INTERVAL_SEC, MAX_ROUNDS=$MAX_ROUNDS(0 means unlimited)" | tee -a "$summary_file"

round=1

is_error_output() {
    local content="$1"

    if grep -Eq '(Traceback|Exception|Error|RemoteDisconnected|Connection reset|Connection aborted|Broken pipe|timed out|Read timed out|Failed to establish|Max retries exceeded|HTTPError)' <<<"$content"; then
        return 0
    fi

    if ! grep -q 'JSON response:' <<<"$content"; then
        return 0
    fi

    return 1
}

while true; do
    if [[ "$MAX_ROUNDS" -gt 0 && "$round" -gt "$MAX_ROUNDS" ]]; then
        echo "Reached MAX_ROUNDS=$MAX_ROUNDS, stop loop." | tee -a "$summary_file"
        break
    fi

    ts="$(date +%Y%m%d_%H%M%S)"
    round_log="$LOG_DIR/round_${round}_${ts}.log"

    echo "==== Round $round @ $(date '+%F %T') ====" | tee -a "$summary_file"

    start_epoch=$(date +%s)
    output="$(${CMD[@]} 2>&1)"
    status=$?
    end_epoch=$(date +%s)
    duration=$((end_epoch - start_epoch))

    printf '%s\n' "$output" > "$round_log"

    echo "round=$round status=$status duration_sec=$duration log=$round_log" | tee -a "$summary_file"

    if [[ $status -ne 0 ]]; then
        echo "[STOP] Non-zero exit code detected." | tee -a "$summary_file"
        exit 1
    fi

    if is_error_output "$output"; then
        echo "[STOP] Error detected from client output." | tee -a "$summary_file"
        echo "Last 10 lines:" | tee -a "$summary_file"
        tail -n 10 "$round_log" | tee -a "$summary_file"
        exit 1
    fi

    latency_line="$(grep -m1 -Eo '[0-9]+\.[0-9]+ seconds' "$round_log" || true)"
    if [[ -n "$latency_line" ]]; then
        echo "analysis: success, latency=$latency_line" | tee -a "$summary_file"
    else
        echo "analysis: success, latency=unknown" | tee -a "$summary_file"
    fi

    ((round++))

    if [[ "$INTERVAL_SEC" -gt 0 ]]; then
        sleep "$INTERVAL_SEC"
    fi
done

echo "Loop finished normally at $(date '+%F %T')" | tee -a "$summary_file"
