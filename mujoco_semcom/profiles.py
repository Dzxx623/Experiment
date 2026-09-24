PAPER_PROFILES = {
    "ideal": dict(delay_ms=0.0, jitter_ms=0.0, loss_pct=0.0),
    "lan": dict(delay_ms=1.0, jitter_ms=0.5, loss_pct=0.0),
    "wifi_good": dict(delay_ms=10.0, jitter_ms=5.0, loss_pct=0.1),
    "wifi_congested": dict(delay_ms=40.0, jitter_ms=20.0, loss_pct=1.0),
    "4g": dict(delay_ms=60.0, jitter_ms=15.0, loss_pct=0.5),
    "poor_4g": dict(delay_ms=150.0, jitter_ms=80.0, loss_pct=3.0),
    "satellite": dict(delay_ms=500.0, jitter_ms=50.0, loss_pct=1.0),
}
REPO_PROFILES = {
    "ideal": dict(delay_ms=0.0, jitter_ms=0.0, loss_pct=0.0),
    "lan": dict(delay_ms=1.0, jitter_ms=0.2, loss_pct=0.0),
    "wifi_good": dict(delay_ms=10.0, jitter_ms=2.0, loss_pct=0.0),
    "wifi_congested": dict(delay_ms=50.0, jitter_ms=10.0, loss_pct=1.0),
    "4g": dict(delay_ms=60.0, jitter_ms=15.0, loss_pct=0.5),
    "poor_4g": dict(delay_ms=200.0, jitter_ms=50.0, loss_pct=2.0),
    "satellite": dict(delay_ms=600.0, jitter_ms=100.0, loss_pct=0.5),
}
def get_profile(name, family="paper"):
    table = PAPER_PROFILES if family == "paper" else REPO_PROFILES
    return dict(table[name])
