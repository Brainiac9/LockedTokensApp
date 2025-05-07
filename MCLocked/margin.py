def get_required_margin(notional, maint_data, log_list=None):
    for tier in maint_data:
        if tier["notionalFloor"] <= notional < tier["notionalCap"]:
            margin = max(notional * tier["maintMarginRate"] - tier["maintAmount"], 0.0)
            return max(margin, notional / 3.0)

    # Log the oversize notional
    max_cap = maint_data[-1]["notionalCap"]
    msg = f"[WARN] Notional {notional:,.2f} exceeds defined margin tiers (cap: {max_cap:,.2f})"
    if log_list is not None:
        log_list.append(msg)

    # Still apply last tier
    last = maint_data[-1]
    margin = max(notional * last["maintMarginRate"] - last["maintAmount"], 0.0)
    return max(margin, notional / 3.0)
