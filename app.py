"""Cycle Hire Dock Recommendation.

Recommends the best Santander Cycles dock to END a ride near a London
destination: where to return the bike, not where to pick one up.

    1. Geocode the destination (and the start, if given):
       postcodes -> postcodes.io, place names -> Nominatim, preferring London.
    2. Fetch live docks around the destination from TfL BikePoint.
    3. Keep docks within 600 m, widening to 1000 m when fewer than three of
       them have a free space.
    4. Estimate ride and walk minutes from straight-line distance x 1.3.
    5. Grade the risk of arriving to a full dock from its free spaces.
    6. Rank on ride + walk + a risk allowance; best is recommended, the next
       two are backups.
    7. Compare with the dock closest to the destination.

Every request goes through `_fetch`, so the tests can replace the network.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode

from pippa_agentapp_sdk import (
    AgentAppError,
    Call,
    Manifest,
    Result,
    load_manifest_file,
    validate_manifest,
)

MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")

# Nominatim answers 403 without a descriptive User-Agent (docs/API_APPS.md).
USER_AGENT = "pippa-agent-app-cycle-dock/0.2 (https://github.com/Pippa-Labs)"

POSTCODES_URL = "https://api.postcodes.io"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
TFL_BIKEPOINT_URL = "https://api.tfl.gov.uk/BikePoint"

# Greater London, roughly. Used to prefer London matches and to spot places
# that are somewhere else.
LONDON_BOX = {"west": -0.5104, "east": 0.3340, "south": 51.2868, "north": 51.6919}
LONDON_VIEWBOX = "-0.5104,51.6919,0.3340,51.2868"

NEAR_RADIUS_M = 600
WIDE_RADIUS_M = 1000
FALLBACK_RADIUS_M = 2000      # only to name the nearest dock with space
ROUTE_FACTOR = 1.3            # straight line -> real streets
RIDE_M_PER_MIN = 250
WALK_M_PER_MIN = 80
LONG_RIDE_MIN = 15            # a ride this long raises the risk one level

RISK_ORDER = ["low", "medium", "high"]
RISK_PENALTY_MIN = {"low": 0, "medium": 3, "high": 8}

# Nominatim's usage policy: at most one request a second.
NOMINATIM_GAP_S = 1.0
# Every request together must finish well inside Pippa's 20-second limit.
# The worst case is four Nominatim lookups (two places, each possibly checked
# twice, a second apart) plus two TfL requests.
TIME_BUDGET_S = 14.0
FETCH_TIMEOUT_S = 4.0

FULL_POSTCODE = re.compile(r"^[A-Z]{1,2}[0-9][A-Z0-9]?\s*[0-9][A-Z]{2}$")
OUTCODE = re.compile(r"^[A-Z]{1,2}[0-9][A-Z0-9]?$")


# ── Errors the person may see ─────────────────────────────────────────────


class NoDestination(AgentAppError):
    user_message = (
        "Tell me where you want to end your ride: a place or postcode in London."
    )
    retryable = False


class PlaceNotFound(AgentAppError):
    user_message = "I could not find that place in London."
    retryable = False


class OutsideLondon(AgentAppError):
    user_message = "Santander Cycles only operate in London."
    retryable = False


class NoDocksNearby(AgentAppError):
    user_message = "There are no Santander Cycles docks within 1 km of there."
    retryable = False


class AllDocksFull(AgentAppError):
    user_message = "Every Santander Cycles dock within 1 km of there is full right now."
    retryable = True


class TflUnavailable(AgentAppError):
    user_message = (
        "TfL's live dock information isn't available right now. "
        "Please try again shortly."
    )
    retryable = True


class Unexpected(AgentAppError):
    user_message = (
        "Something went wrong working out the best dock. Please try again shortly."
    )
    retryable = True


class LookupUnavailable(AgentAppError):
    user_message = (
        "The place lookup service isn't responding right now. "
        "Please try again shortly."
    )
    retryable = True


def _with_message(error: AgentAppError, message: str) -> AgentAppError:
    error.user_message = message
    return error


# ── Small value types ─────────────────────────────────────────────────────


@dataclass
class Place:
    lat: float
    lon: float
    label: str


@dataclass
class Dock:
    name: str
    lat: float
    lon: float
    empty: int
    total: int
    distance_m: float = 0.0          # straight line, dock -> destination


@dataclass
class Scored:
    dock: Dock
    risk: str                         # low / medium / high / full
    ride_min: int | None
    walk_min: int
    walk_m: float
    total_min: int | None             # None when full
    sort_key: tuple = ()


# ── Pure helpers (easy to test) ───────────────────────────────────────────


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def in_london(lat: float, lon: float) -> bool:
    box = LONDON_BOX
    return box["south"] <= lat <= box["north"] and box["west"] <= lon <= box["east"]


def base_risk(empty: int) -> str:
    if empty <= 0:
        return "full"
    if empty <= 2:
        return "high"
    if empty <= 5:
        return "medium"
    return "low"


def raise_risk(risk: str) -> str:
    if risk == "full":
        return risk
    return RISK_ORDER[min(RISK_ORDER.index(risk) + 1, len(RISK_ORDER) - 1)]


def whole(x: float) -> int:
    """Round half up, so 2.5 minutes is 3 rather than banker's 2."""
    return int(math.floor(x + 0.5))


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def _to_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def score(dock: Dock, origin: Place | None) -> Scored:
    walk_m = dock.distance_m * ROUTE_FACTOR
    walk = walk_m / WALK_M_PER_MIN
    risk = base_risk(dock.empty)
    ride = None
    if origin is not None:
        ride = haversine_m(origin.lat, origin.lon, dock.lat, dock.lon) * ROUTE_FACTOR / RIDE_M_PER_MIN
        if ride >= LONG_RIDE_MIN:
            risk = raise_risk(risk)
    ride_r = whole(ride) if ride is not None else None
    walk_r = whole(walk)
    total = None
    if risk != "full":
        total = (ride_r or 0) + walk_r + RISK_PENALTY_MIN[risk]
    exact = (ride or 0) + walk + RISK_PENALTY_MIN.get(risk, 0)
    return Scored(dock, risk, ride_r, walk_r, walk_m, total,
                  (total if total is not None else 10**9, exact, dock.distance_m))


def why_not(nearest: Scored, best: Scored) -> str:
    if nearest.risk == "full":
        space = "it is full, with no free spaces"
    elif nearest.dock.empty == 1:
        space = f"only 1 free space, {nearest.risk} risk"
    else:
        space = f"only {nearest.dock.empty} free spaces, {nearest.risk} risk"
    if nearest.risk == "full":
        return space
    reasons = [space] if nearest.risk != "low" else []
    if (nearest.ride_min is not None and best.ride_min is not None
            and nearest.ride_min > best.ride_min):
        reasons.append(f"{nearest.ride_min - best.ride_min} min longer ride")
    slower = (nearest.total_min or 0) - (best.total_min or 0)
    if slower > 0:
        reasons.append(f"{slower} min slower overall")
    elif not reasons:
        reasons.append("no quicker overall once ride time and risk are counted")
    return "; ".join(reasons)


# ── The app ───────────────────────────────────────────────────────────────


class App:
    def __init__(self) -> None:
        self._manifest = validate_manifest(load_manifest_file(MANIFEST_PATH))
        # Instance attributes so tests can replace them without waiting.
        self._sleep = time.sleep
        self._clock = time.monotonic
        self._last_nominatim: float | None = None
        self._started = 0.0

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    # The one method that touches the network.
    def _fetch(self, url: str, timeout: float = FETCH_TIMEOUT_S) -> bytes:
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def _timeout(self, unavailable: type[AgentAppError]) -> float:
        left = TIME_BUDGET_S - (self._clock() - self._started)
        if left < 1.0:
            raise unavailable("ran out of time before the next request")
        return min(FETCH_TIMEOUT_S, left)

    def _get_json(self, url: str, unavailable: type[AgentAppError],
                  not_found: AgentAppError | None = None):
        try:
            raw = self._fetch(url, timeout=self._timeout(unavailable))
        except urllib.error.HTTPError as exc:          # before URLError: it is one
            if exc.code in (400, 404) and not_found is not None:
                raise not_found
            raise unavailable(f"upstream answered {exc.code}")
        except TimeoutError as exc:                    # not a URLError
            raise unavailable(f"timed out: {exc}")
        except (urllib.error.URLError, OSError) as exc:   # DNS, refused, TLS
            raise unavailable(f"could not connect: {type(exc).__name__}: {exc}")
        except (http.client.HTTPException, ValueError) as exc:
            # A connection cut mid-answer (IncompleteRead, BadStatusLine) is
            # neither a URLError nor an OSError.
            raise unavailable(f"broken response: {type(exc).__name__}: {exc}")
        try:
            return json.loads(raw)
        except (ValueError, TypeError, RecursionError) as exc:
            raise unavailable(f"response was not JSON: {type(exc).__name__}")

    # ── handle ────────────────────────────────────────────────────────────

    def handle(self, call: Call) -> Result:
        if call.outcome_id != "dock.recommend":
            raise AgentAppError(f"this app cannot do {call.outcome_id!r}")
        self._started = self._clock()
        self._last_nominatim = None
        try:
            return self._recommend(call)
        except AgentAppError:
            raise
        except Exception as exc:
            # A bug or a reply shaped in a way nobody expected. Say something
            # true rather than letting the call die with no explanation.
            raise Unexpected(f"{type(exc).__name__}: {exc}") from exc

    def _recommend(self, call: Call) -> Result:
        now: datetime = call.context.now

        destination_text = " ".join((call.inputs.get("destination") or "").split())
        origin_text = " ".join((call.inputs.get("origin") or "").split())
        if not destination_text:
            raise NoDestination("no destination in the message")

        destination = self._geocode(destination_text)

        notes: list[str] = []
        origin: Place | None = None
        if origin_text:
            try:
                origin = self._geocode(origin_text)
            except (PlaceNotFound, OutsideLondon):
                notes.append(
                    f"I couldn't place where you're starting from ({origin_text}), "
                    "so this is ranked on walking time only"
                )
        else:
            notes.append(
                "Tell me where you're starting from for a better estimate; "
                "this is ranked on walking time only"
            )

        docks = self._docks_near(destination, WIDE_RADIUS_M)
        if not docks:
            raise _with_message(
                NoDocksNearby(f"no docks within {WIDE_RADIUS_M} m of {destination_text!r}"),
                f"There are no Santander Cycles docks within 1 km of {destination_text}.",
            )

        near = [d for d in docks if d.distance_m <= NEAR_RADIUS_M]
        if sum(1 for d in near if d.empty > 0) >= 3:
            pool = near
        else:
            pool = docks
        scored = [score(d, origin) for d in pool]
        usable = sorted((s for s in scored if s.risk != "full"), key=lambda s: s.sort_key)

        if not usable:
            raise self._all_full(destination, destination_text)

        best, backups = usable[0], usable[1:3]
        nearest = min(scored, key=lambda s: s.dock.distance_m)

        data: dict = {}
        self._put(data, "recommended", best)
        for i, backup in enumerate(backups, start=1):
            self._put(data, f"backup{i}", backup)

        data["nearest_is_recommended"] = nearest.dock is best.dock
        if nearest.dock is not best.dock:
            data["nearest_name"] = nearest.dock.name
            data["nearest_empty_docks"] = nearest.dock.empty
            data["nearest_total_docks"] = nearest.dock.total
            data["nearest_risk"] = nearest.risk
            data["nearest_metres_closer"] = max(
                0, int(round((best.walk_m - nearest.walk_m) / 10.0)) * 10
            )
            data["nearest_walk_min_closer"] = max(0, best.walk_min - nearest.walk_min)
            data["nearest_why_not"] = why_not(nearest, best)

        # Every value in `data` is a plain string, number or boolean, so the
        # answer always turns into JSON cleanly.
        note = (
            f"Live TfL dock data checked at {self._clock_time(now)}; "
            "spaces can change before you arrive."
        )
        if data["nearest_is_recommended"]:
            note += " The recommended dock is also the closest one to your destination."
        for extra in notes:
            note += f" {extra}."
        data["note"] = note

        return Result(data=data, narrative=self._sentence(best, destination_text))

    # ── geocoding ─────────────────────────────────────────────────────────

    def _geocode(self, text: str) -> Place:
        compact = text.upper().strip()
        if FULL_POSTCODE.match(compact):
            return self._postcode(text, compact)
        if OUTCODE.match(compact):
            return self._outcode(text, compact)
        return self._place_name(text)

    def _postcode(self, text: str, compact: str) -> Place:
        not_found = _with_message(
            PlaceNotFound(f"postcodes.io has no {compact!r}"),
            f"I could not find the postcode {text.upper()}.",
        )
        payload = self._get_json(
            f"{POSTCODES_URL}/postcodes/{quote(compact, safe='')}",
            LookupUnavailable, not_found,
        )
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or result.get("latitude") is None:
            raise not_found
        lat, lon = float(result["latitude"]), float(result["longitude"])
        region = result.get("region")
        outside = (region != "London") if region else not in_london(lat, lon)
        if outside:
            raise _with_message(
                OutsideLondon(f"{compact!r} is in {region!r}"),
                f"Santander Cycles only operate in London, and {text.upper()} is not in London.",
            )
        return Place(lat, lon, text.upper())

    def _outcode(self, text: str, compact: str) -> Place:
        not_found = _with_message(
            PlaceNotFound(f"postcodes.io has no outcode {compact!r}"),
            f"I could not find the postcode area {compact}.",
        )
        payload = self._get_json(
            f"{POSTCODES_URL}/outcodes/{quote(compact, safe='')}",
            LookupUnavailable, not_found,
        )
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or result.get("latitude") is None:
            raise not_found
        lat, lon = float(result["latitude"]), float(result["longitude"])
        if not in_london(lat, lon):
            raise _with_message(
                OutsideLondon(f"outcode {compact!r} is outside London"),
                f"Santander Cycles only operate in London, and {compact} is not in London.",
            )
        return Place(lat, lon, compact)

    def _nominatim(self, params: dict) -> list:
        if self._last_nominatim is not None:
            wait = NOMINATIM_GAP_S - (self._clock() - self._last_nominatim)
            if wait > 0:
                self._sleep(wait)
        query = urlencode({"format": "jsonv2", "limit": 1, "countrycodes": "gb", **params})
        try:
            payload = self._get_json(f"{NOMINATIM_URL}?{query}", LookupUnavailable)
        finally:
            self._last_nominatim = self._clock()
        return payload if isinstance(payload, list) else []

    def _place_name(self, text: str) -> Place:
        not_found = _with_message(
            PlaceNotFound(f"nominatim found nothing for {text!r}"),
            f"I could not find {text} in London.",
        )
        outside = _with_message(
            OutsideLondon(f"{text!r} is outside London"),
            f"Santander Cycles only operate in London, and {text} is not in London.",
        )
        # Prefer London, but do not insist: a hard London-only search turns
        # "manchester" into Manchester Square, W1, without a word.
        found = self._nominatim({"q": text, "viewbox": LONDON_VIEWBOX, "bounded": 0})
        if not found:
            raise not_found
        top = found[0]
        try:
            lat, lon = float(top["lat"]), float(top["lon"])
        except (KeyError, TypeError, ValueError):
            raise not_found
        if not in_london(lat, lon):
            raise outside

        # The London match is named something else ("Manchester Square" for
        # "manchester"). If the plain search finds a town or city of exactly
        # that name outside London, that is what they meant.
        if _normalise(str(top.get("name", ""))) != _normalise(text):
            plain = self._nominatim({"q": text})
            if plain:
                other = plain[0]
                try:
                    o_lat, o_lon = float(other["lat"]), float(other["lon"])
                    rank = int(other.get("place_rank", 99))
                except (KeyError, TypeError, ValueError):
                    o_lat = o_lon = None
                    rank = 99
                if (o_lat is not None and not in_london(o_lat, o_lon) and rank <= 16
                        and _normalise(str(other.get("name", ""))) == _normalise(text)):
                    raise outside
        return Place(lat, lon, text)

    # ── docks ─────────────────────────────────────────────────────────────

    def _docks_near(self, place: Place, radius: int) -> list[Dock]:
        query = urlencode({"lat": f"{place.lat:.5f}", "lon": f"{place.lon:.5f}", "radius": radius})
        payload = self._get_json(f"{TFL_BIKEPOINT_URL}?{query}", TflUnavailable)
        if not isinstance(payload, dict) or not isinstance(payload.get("places"), list):
            raise TflUnavailable("BikePoint answer had no list of places")
        docks = []
        for item in payload["places"]:
            dock = self._dock(item, place)
            if dock is not None and dock.distance_m <= radius:
                docks.append(dock)
        return docks

    @staticmethod
    def _dock(item, place: Place) -> Dock | None:
        if not isinstance(item, dict):
            return None
        props: dict = {}
        for prop in item.get("additionalProperties") or []:
            if isinstance(prop, dict) and "key" in prop:
                props[prop["key"]] = prop.get("value")
        # A dock that is locked or not installed cannot take a bike.
        if str(props.get("Installed", "true")).lower() != "true":
            return None
        if str(props.get("Locked", "false")).lower() == "true":
            return None
        empty, total = _to_int(props.get("NbEmptyDocks")), _to_int(props.get("NbDocks"))
        try:
            lat, lon = float(item["lat"]), float(item["lon"])
        except (KeyError, TypeError, ValueError):
            return None
        if empty is None or total is None:
            return None
        name = str(item.get("commonName") or "Unnamed dock")
        return Dock(name, lat, lon, max(0, empty), total,
                    haversine_m(place.lat, place.lon, lat, lon))

    def _all_full(self, destination: Place, text: str) -> AgentAppError:
        error = AllDocksFull(f"every dock within {WIDE_RADIUS_M} m of {text!r} is full")
        message = f"Every Santander Cycles dock within 1 km of {text} is full right now."
        try:
            wider = self._docks_near(destination, FALLBACK_RADIUS_M)
        except AgentAppError:
            wider = []
        with_space = sorted((d for d in wider if d.empty > 0), key=lambda d: d.distance_m)
        if with_space:
            d = with_space[0]
            spaces = "1 free space" if d.empty == 1 else f"{d.empty} free spaces"
            message += (
                f" The nearest dock with space is {d.name}, about "
                f"{int(round(d.distance_m * ROUTE_FACTOR / 10.0)) * 10} m walk away ({spaces})."
            )
        else:
            message += " None within 2 km has space either; please try again shortly."
        return _with_message(error, message)

    # ── output helpers ────────────────────────────────────────────────────

    @staticmethod
    def _put(data: dict, prefix: str, s: Scored) -> None:
        data[f"{prefix}_name"] = s.dock.name
        data[f"{prefix}_empty_docks"] = s.dock.empty
        data[f"{prefix}_total_docks"] = s.dock.total
        data[f"{prefix}_risk"] = s.risk
        if s.ride_min is not None:
            data[f"{prefix}_ride_min"] = s.ride_min
        data[f"{prefix}_walk_min"] = s.walk_min
        data[f"{prefix}_total_min"] = s.total_min

    @staticmethod
    def _clock_time(moment: datetime) -> str:
        try:
            from zoneinfo import ZoneInfo

            local = moment.astimezone(ZoneInfo("Europe/London"))
        except Exception:                      # no time zone data: say UTC
            return moment.astimezone(timezone.utc).strftime("%H:%M UTC")
        return local.strftime("%H:%M London time")

    @staticmethod
    def _sentence(best: Scored, destination: str) -> str:
        spaces = "1 free space" if best.dock.empty == 1 else f"{best.dock.empty} free spaces"
        trip = f"about {best.walk_min} min walk to {destination}"
        if best.ride_min is not None:
            trip = f"about {best.ride_min} min ride, then {best.walk_min} min walk to {destination}"
        return (
            f"Best Santander Cycles dock: {best.dock.name} "
            f"({spaces} of {best.dock.total}, {best.risk} risk), {trip}."
        )
