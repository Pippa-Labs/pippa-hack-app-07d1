"""Tests for Cycle Hire Dock Recommendation.

The build runs these on every push and will not publish if one fails. They
never touch the network (conftest.py blocks it): every request goes through
`App._fetch`, which each test replaces with a fake that answers by URL.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timezone

import pytest
from pippa_agentapp_sdk import LocalRunner

import app as app_module
from app import (
    AllDocksFull,
    App,
    LookupUnavailable,
    NoDestination,
    NoDocksNearby,
    OutsideLondon,
    PlaceNotFound,
    TflUnavailable,
    base_risk,
    raise_risk,
)

NOW = datetime(2026, 11, 20, 15, 10, tzinfo=timezone.utc)
FRESH = "2026-11-20T15:07:23.17Z"        # TfL's real timestamp format
OLD = "2026-11-20T14:40:00Z"             # 30 minutes before NOW

# South Kensington, SW7 2AZ (real postcodes.io coordinates).
DEST = (51.499334, -0.179178)
M_PER_DEG_LAT = 111_195.0

# One dock record exactly as TfL BikePoint returned it on 20 November 2026,
# with only the "$type"/"url"/"children" fields trimmed. The app must read
# NbEmptyDocks and NbDocks: here NbBikes + NbEmptyDocks (13) happens to equal
# NbDocks, but at broken docks it does not.
REAL_RECORD = json.loads("""
{"id": "BikePoints_432", "commonName": "Exhibition Road Museums 1, South Kensington",
 "lat": 51.496468, "lon": -0.173796, "distance": 171.08253156968385, "placeType": "BikePoint",
 "additionalProperties": [
  {"key": "TerminalName", "value": "022158", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "Installed", "value": "true", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "Locked", "value": "false", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "InstallDate", "value": "1319448720000", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "RemovalDate", "value": "", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "Temporary", "value": "false", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "NbBikes", "value": "8", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "NbEmptyDocks", "value": "5", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "NbDocks", "value": "13", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "NbStandardBikes", "value": "7", "modified": "2026-11-20T15:07:23.17Z"},
  {"key": "NbEBikes", "value": "1", "modified": "2026-11-20T15:07:23.17Z"}]}
""")


# ── Building fake answers ─────────────────────────────────────────────────


def dock(name, metres_north, empty, total, *, bikes=None, modified=FRESH,
         locked="false", installed="true", centre=DEST):
    """A TfL BikePoint place, `metres_north` of the destination."""
    bikes = total - empty - 1 if bikes is None else bikes   # one broken dock
    props = {
        "TerminalName": "000000", "Installed": installed, "Locked": locked,
        "Temporary": "false", "NbBikes": str(bikes),
        "NbEmptyDocks": str(empty), "NbDocks": str(total),
    }
    return {
        "$type": "Tfl.Api.Presentation.Entities.Place, Tfl.Api.Presentation.Entities",
        "id": f"BikePoints_{abs(hash(name)) % 1000}",
        "commonName": name,
        "lat": centre[0] + metres_north / M_PER_DEG_LAT,
        "lon": centre[1],
        "placeType": "BikePoint",
        "additionalProperties": [
            {"key": k, "value": v, "modified": modified} for k, v in props.items()
        ],
    }


def bikepoint(*places):
    return json.dumps({"centrePoint": [51.499, -0.179], "places": list(places)}).encode()


def postcode(lat, lon, region="London"):
    return json.dumps({"status": 200, "result": {
        "postcode": "SW7 2AZ", "latitude": lat, "longitude": lon,
        "region": region, "country": "England"}}).encode()


def nominatim(name, lat, lon, rank=30):
    return json.dumps([{"name": name, "lat": str(lat), "lon": str(lon),
                        "place_rank": rank, "display_name": name}]).encode()


def http_error(url, code):
    return urllib.error.HTTPError(url, code, "error", {}, io.BytesIO(b"{}"))


class Fake:
    """Answers each request by the first rule whose text is in the URL."""

    def __init__(self, rules):
        self.rules = rules
        self.urls: list[str] = []

    def __call__(self, url, timeout=6.0):
        self.urls.append(url)
        for needle, answer in self.rules:
            if needle in url:
                if isinstance(answer, BaseException):
                    raise answer
                if callable(answer):
                    return answer(url)
                return answer
        raise AssertionError(f"no fake answer for {url}")


def run(inputs, rules, *, explicit=False, now=NOW):
    instance = App()
    fake = Fake(rules)
    instance._fetch = fake
    instance._sleep = lambda seconds: None
    result = LocalRunner(instance).call(
        "dock.recommend", inputs, granted_secrets={}, now=now,
        request_id="t1", explicit=explicit,
    ).result
    return result, fake


SW7 = ("api.postcodes.io/postcodes/SW7", postcode(*DEST))

# A typical South Kensington picture: plenty of docks within 600 m.
NEARBY = bikepoint(
    dock("Nearest, 1 space", 50, 1, 14),         # closest, but high risk
    dock("Roomy", 200, 9, 20),                   # low risk
    dock("Medium", 300, 4, 18),                  # medium risk
    dock("Far roomy", 550, 12, 30),              # low risk, more walking
    dock("Full one", 100, 0, 16),                # never recommended
)


# ── The manifest ──────────────────────────────────────────────────────────


def test_the_manifest_is_valid():
    instance = App()
    assert instance.manifest.app_id == "hack.app_07d1"
    assert [o.id for o in instance.manifest.outcomes] == ["dock.recommend"]


# ── Happy paths ───────────────────────────────────────────────────────────


def test_recommends_a_dock_and_two_backups_without_a_start():
    result, _ = run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", NEARBY)])
    data = result.data
    # walk: 200 m x 1.3 / 80 = 3.25 -> 3 min, low risk, no ride
    assert data["recommended_name"] == "Roomy"
    assert data["recommended_empty_docks"] == 9
    assert data["recommended_total_docks"] == 20
    assert data["recommended_risk"] == "low"
    assert data["recommended_walk_min"] == 3
    assert data["recommended_total_min"] == 3
    assert "recommended_ride_min" not in data
    # Medium: 300 m -> 5 min + 3 penalty = 8. Far roomy: 550 m -> 9 min + 0 = 9.
    # Nearest (1 space, high): 1 min + 8 = 9; ties broken on exact minutes.
    assert data["backup1_name"] == "Medium"
    assert data["backup1_total_min"] == 8
    assert {data["backup2_name"]} <= {"Far roomy", "Nearest, 1 space"}
    assert "starting from" in data["note"]
    assert "spaces can change before you arrive" in data["note"]


def test_the_nearest_dock_is_compared_even_when_full():
    result, _ = run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", NEARBY)])
    data = result.data
    assert data["nearest_is_recommended"] is False
    assert data["nearest_name"] == "Nearest, 1 space"
    assert data["nearest_empty_docks"] == 1
    assert data["nearest_risk"] == "high"
    assert data["nearest_metres_closer"] in (190, 200)   # (200 - 50) x 1.3 = about 195
    assert data["nearest_walk_min_closer"] == 2          # 3 min vs 1 min
    assert "only 1 free space, high risk" in data["nearest_why_not"]


def test_a_full_nearest_dock_is_shown_as_full():
    docks = bikepoint(dock("Full and closest", 30, 0, 16),
                      dock("A", 200, 8, 20), dock("B", 250, 7, 20), dock("C", 300, 9, 20))
    data = run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", docks)])[0].data
    assert data["nearest_name"] == "Full and closest"
    assert data["nearest_risk"] == "full"
    assert "full" in data["nearest_why_not"]
    for key in ("recommended_name", "backup1_name", "backup2_name"):
        assert data[key] != "Full and closest"


def test_when_the_nearest_is_recommended_it_says_so():
    docks = bikepoint(dock("Close and roomy", 60, 10, 20), dock("B", 300, 9, 20),
                      dock("C", 400, 9, 20))
    data = run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", docks)])[0].data
    assert data["nearest_is_recommended"] is True
    assert "nearest_name" not in data
    assert "also the closest" in data["note"]


def test_with_a_start_it_adds_ride_time():
    # Origin 1000 m south of the destination: rides are 1300-1560 m / 250.
    origin = (DEST[0] - 1000 / M_PER_DEG_LAT, DEST[1])
    rules = [SW7, ("postcodes/SW1", postcode(*origin)), ("BikePoint", NEARBY)]
    data = run({"destination": "SW7 2AZ", "origin": "SW1A 1AA"}, rules)[0].data
    assert data["recommended_name"] == "Roomy"
    assert data["recommended_ride_min"] == 6             # 1200 m x 1.3 / 250 = 6.24
    assert data["recommended_total_min"] == 6 + 3 + 0
    assert "starting from" not in data["note"]


def test_a_long_ride_raises_the_risk_one_level():
    assert base_risk(0) == "full"
    assert base_risk(2) == "high"
    assert base_risk(5) == "medium"
    assert base_risk(6) == "low"
    assert raise_risk("low") == "medium"
    assert raise_risk("high") == "high"
    assert raise_risk("full") == "full"
    # 4 km away: 4000 x 1.3 / 250 = 20.8 min ride, so 9 spaces becomes medium.
    origin = (DEST[0] - 4000 / M_PER_DEG_LAT, DEST[1])
    rules = [SW7, ("postcodes/SW1", postcode(*origin)), ("BikePoint", NEARBY)]
    data = run({"destination": "SW7 2AZ", "origin": "SW1A 1AA"}, rules)[0].data
    assert data["recommended_risk"] in ("medium", "high")
    assert data["recommended_ride_min"] >= 15


def test_it_widens_to_1000m_when_fewer_than_three_near_docks_have_space():
    docks = bikepoint(dock("Near full", 100, 0, 15), dock("Near ok", 300, 7, 20),
                      dock("Wide one", 800, 10, 25), dock("Wide two", 900, 12, 25))
    data = run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", docks)])[0].data
    names = {data["recommended_name"], data["backup1_name"], data["backup2_name"]}
    assert names == {"Near ok", "Wide one", "Wide two"}


def test_docks_beyond_1000m_are_ignored_and_backups_are_optional():
    docks = bikepoint(dock("Only one", 300, 7, 20), dock("Too far", 1200, 20, 30))
    data = run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", docks)])[0].data
    assert data["recommended_name"] == "Only one"
    assert "backup1_name" not in data


def test_locked_docks_are_skipped():
    docks = bikepoint(dock("Locked", 50, 10, 20, locked="true"), dock("Open", 200, 8, 20),
                      dock("B", 300, 8, 20), dock("C", 400, 8, 20))
    data = run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", docks)])[0].data
    assert "Locked" not in json.dumps(data, default=str)


def test_old_data_is_flagged_as_possibly_stale():
    docks = bikepoint(dock("Roomy", 200, 9, 20, modified=OLD), dock("B", 300, 9, 20),
                      dock("C", 400, 9, 20))
    data = run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", docks)])[0].data
    assert data["data_possibly_stale"] is True
    assert data["data_updated"] == datetime(2026, 11, 20, 14, 40, tzinfo=timezone.utc)
    assert "over 15 minutes old" in data["note"]


def test_fresh_data_is_not_flagged():
    data = run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", NEARBY)])[0].data
    assert data["data_possibly_stale"] is False
    assert data["data_updated"].utcoffset() is not None


def test_the_real_tfl_record_is_read_correctly():
    """Parsing checked against a real BikePoint answer, not one we invented."""
    centre = (51.4988, -0.1749)       # the point that real query was made from
    rules = [("postcodes/SW7", postcode(*centre)),
             ("BikePoint", bikepoint(REAL_RECORD))]
    data = run({"destination": "SW7 2AZ"}, rules, now=datetime(2026, 11, 20, 15, 10, tzinfo=timezone.utc))[0].data
    assert data["recommended_name"] == "Exhibition Road Museums 1, South Kensington"
    assert data["recommended_empty_docks"] == 5      # NbEmptyDocks
    assert data["recommended_total_docks"] == 13     # NbDocks, not NbDocks - NbBikes
    assert data["recommended_risk"] == "medium"
    assert data["recommended_walk_min"] == 4         # ~270 m straight x 1.3 / 80 = 4.4
    assert data["nearest_is_recommended"] is True


def test_place_names_use_nominatim_with_a_user_agent_and_london_preference():
    rules = [("nominatim", nominatim("Waterloo", *DEST)), ("BikePoint", NEARBY)]
    result, fake = run({"destination": "waterloo"}, rules)
    assert result.data["recommended_name"] == "Roomy"
    url = fake.urls[0]
    assert "viewbox=" in url and "countrycodes=gb" in url
    assert app_module.USER_AGENT.startswith("pippa-agent-app-")


def test_a_london_match_with_another_name_stays_in_london():
    """"kings cross" also names a hamlet in Buckinghamshire; London wins."""
    rules = [
        ("bounded=0", nominatim("London King's Cross", *DEST)),
        ("nominatim", nominatim("Kings Cross", 51.7854, -0.9269, rank=25)),
        ("BikePoint", NEARBY),
    ]
    result, _ = run({"destination": "kings cross"}, rules)
    assert result.data["recommended_name"] == "Roomy"


def test_picked_by_name_answers_the_same():
    rules = [SW7, ("BikePoint", NEARBY)]
    assert run({"destination": "SW7 2AZ"}, rules, explicit=True)[0].data == \
        run({"destination": "SW7 2AZ"}, rules)[0].data


# ── Bad input ─────────────────────────────────────────────────────────────


def test_no_destination_says_what_to_type():
    with pytest.raises(NoDestination) as failure:
        run({"destination": ""}, [])
    assert "place or postcode" in failure.value.user_message


def test_an_unknown_postcode_is_named():
    with pytest.raises(PlaceNotFound) as failure:
        run({"destination": "ZZ9 9ZZ"},
            [("postcodes/ZZ9", http_error("https://api.postcodes.io", 404))])
    assert "ZZ9 9ZZ" in failure.value.user_message


def test_a_postcode_outside_london_is_refused():
    with pytest.raises(OutsideLondon) as failure:
        run({"destination": "M1 1AE"},
            [("postcodes/M1", postcode(53.48, -2.23, region="North West"))])
    assert "only operate in London" in failure.value.user_message


def test_nominatims_empty_answer_is_place_not_found():
    """Nominatim answers 200 with [] for a place that does not exist."""
    with pytest.raises(PlaceNotFound) as failure:
        run({"destination": "narnia castle"}, [("nominatim", b"[]")])
    assert "narnia castle" in failure.value.user_message


def test_a_place_outside_london_is_refused():
    with pytest.raises(OutsideLondon):
        run({"destination": "brighton"},
            [("nominatim", nominatim("Brighton", 50.82, -0.14, rank=16))])


def test_manchester_is_not_turned_into_manchester_square():
    """A London-preferring search answers "Manchester Square, W1"."""
    rules = [
        ("bounded=0", nominatim("Manchester Square", 51.5167, -0.1526, rank=25)),
        ("nominatim", nominatim("Manchester", 53.4424, -2.2324, rank=16)),
    ]
    with pytest.raises(OutsideLondon):
        run({"destination": "manchester"}, rules)


def test_an_unknown_start_falls_back_to_walking_only():
    rules = [SW7, ("nominatim", b"[]"), ("BikePoint", NEARBY)]
    data = run({"destination": "SW7 2AZ", "origin": "nowhere-ville"}, rules)[0].data
    assert "recommended_ride_min" not in data
    assert "nowhere-ville" in data["note"]


# ── When TfL or the lookups misbehave ─────────────────────────────────────


def test_no_docks_within_1km():
    with pytest.raises(NoDocksNearby) as failure:
        run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", bikepoint())])
    assert "within 1 km" in failure.value.user_message


def test_all_nearby_full_names_the_nearest_dock_with_space():
    def answer(url):
        if "radius=1000" in url:
            return bikepoint(dock("Full A", 100, 0, 15), dock("Full B", 400, 0, 20))
        return bikepoint(dock("Full A", 100, 0, 15), dock("Has room", 1500, 6, 20))
    with pytest.raises(AllDocksFull) as failure:
        run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", answer)])
    assert "Has room" in failure.value.user_message
    assert "6 free spaces" in failure.value.user_message


@pytest.mark.parametrize("answer", [
    http_error("https://api.tfl.gov.uk/BikePoint", 429),
    http_error("https://api.tfl.gov.uk/BikePoint", 500),
    TimeoutError("timed out"),
    urllib.error.URLError("no route"),
    b"not json",
    b"{}",                                 # 200 with no places at all
    b"[]",
])
def test_tfl_failures_say_try_again_shortly(answer):
    with pytest.raises(TflUnavailable) as failure:
        run({"destination": "SW7 2AZ"}, [SW7, ("BikePoint", answer)])
    assert failure.value.retryable is True
    assert "try again shortly" in failure.value.user_message


@pytest.mark.parametrize("answer", [
    http_error("https://nominatim.openstreetmap.org", 429),
    TimeoutError("timed out"),
])
def test_lookup_failures_say_try_again_shortly(answer):
    with pytest.raises(LookupUnavailable) as failure:
        run({"destination": "waterloo"}, [("nominatim", answer)])
    assert failure.value.retryable is True


def test_postcode_lookup_timeout_is_caught():
    with pytest.raises(LookupUnavailable):
        run({"destination": "SW7 2AZ"}, [("postcodes", TimeoutError("slow"))])
