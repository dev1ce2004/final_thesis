"""MCDM-driven Terran marine push for PySC2.

Terran vs Zerg (Easy) on Simple64, RAW action space.

Architecture
------------
MCDM  -> chooses the strategic priority (which *kind* of thing to do now)
Agent -> executes it, and enforces hard RTS constraints MCDM cannot violate

Hard constraints (checked before MCDM every step):
  1. never let supply block production
  2. never ignore enemies inside our own base

Everything else is dispatched from the MCDM decision, with a fallback
chain so that a step is never wasted when the chosen action has nothing
to do. That keeps the invariant from the original design - MCDM can
never *prevent* an attack - while still letting MCDM actually steer.

Performance
-----------
The observation is read ONCE per step into a Snapshot (bucketed by
alliance and unit type, indexed by tag). Every query afterwards is a
dict/list lookup instead of another linear scan of raw_units. This
matters far more in a real game than it looks here: pysc2 returns
raw_units as a NamedNumpyArray whose attribute access goes through
__getattr__ on every single field read, so the old "scan raw_units
seventeen times per step" pattern paid that cost thousands of times per
step. Where available, the snapshot is built from a single
ndarray.tolist() call, which converts the whole observation in C.
"""

from __future__ import annotations

import math
import csv
import os
from collections import Counter

import numpy as np
from absl import app, flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

FLAGS = flags.FLAGS


def _flag(define, name, default, description):
    # Guarded so the module can also be imported by a batch runner that
    # already defines flags, instead of raising DuplicateFlagError.
    if name not in FLAGS:
        define(name, default, description)


_flag(flags.DEFINE_integer, "episodes", 1, "Number of episodes to play.")
_flag(flags.DEFINE_boolean, "visualize", False, "Render the pysc2 visualizer.")
_flag(flags.DEFINE_string, "difficulty", "easy", "Built-in bot difficulty.")
_flag(flags.DEFINE_integer, "step_mul", 8, "Game loops per agent step.")
_flag(flags.DEFINE_boolean, "quiet", False, "Suppress per-action logging.")
_flag(flags.DEFINE_boolean, "idle_speedup", True,
      "Advance more game loops per step while nothing needs doing.")
_flag(flags.DEFINE_string, "mcdm", "saw",
      "Aggregation method: saw, topsis, vikor, or none (ablation).")
_flag(flags.DEFINE_string, "normalization", "classic",
      "classic = SAW/VIKOR linear, TOPSIS vector; shared = linear for all.")
_flag(flags.DEFINE_float, "vikor_v", 0.5, "VIKOR weight of group utility.")
_flag(flags.DEFINE_string, "decision_log", "",
      "CSV path for per-decision matrices, scores and rankings.")
_flag(flags.DEFINE_string, "map", "Simple64", "Map name or .SC2Map file name.")
_flag(flags.DEFINE_string, "map_dir", "",
      "Subfolder of StarCraft II/Maps holding the map, if auto-detection fails.")
_flag(flags.DEFINE_integer, "raw_resolution", 64,
      "Raw coordinate grid. Positions are scaled into this grid, so the "
      "agent calibrates its own distances against it at game start.")
_flag(flags.DEFINE_string, "profile", "",
      "Build profile: easy, medium, hard. Empty = follow --difficulty.")
_flag(flags.DEFINE_integer, "random_seed", -1,
      "SC2 random seed; use the same value across methods for paired runs.")


# ======================================================================
# CONSTANTS
# ======================================================================

RAW_RESOLUTION = 64

SCV_COST = 50
MARINE_COST = 50
DEPOT_COST = 100
BARRACKS_COST = 150
ORBITAL_COST = 150
MULE_ENERGY = 50

SCV = int(units.Terran.SCV)
CC = int(units.Terran.CommandCenter)
ORBITAL = int(units.Terran.OrbitalCommand)
PLANETARY = int(units.Terran.PlanetaryFortress)

BARRACKS = int(units.Terran.Barracks)
BARRACKS_FLYING = int(units.Terran.BarracksFlying)

DEPOT = int(units.Terran.SupplyDepot)
DEPOT_LOWERED = int(units.Terran.SupplyDepotLowered)

MARINE = int(units.Terran.Marine)
BUNKER = int(units.Terran.Bunker)

BUNKER_COST = 100
CC_COST = 400

BASE_IDS = (CC, ORBITAL, PLANETARY)
DEPOT_IDS = (DEPOT, DEPOT_LOWERED)
BARRACKS_IDS = (BARRACKS, BARRACKS_FLYING)

# PlayerRelative in the raw observation.
ALLIANCE_SELF = 1
ALLIANCE_NEUTRAL = 3
ALLIANCE_ENEMY = 4

# Neutral resources - buildings cannot be placed on top of these.
MINERAL_IDS = frozenset(
    {146, 147, 341, 483, 665, 666, 796, 797, 884, 885, 886, 887, 1961})
GEYSER_IDS = frozenset({342, 343, 344, 608, 880, 881, 1962})
RESOURCE_IDS = MINERAL_IDS | GEYSER_IDS

# Zerg units/structures that exert no combat pressure. Used for threat
# assessment only - they are still legal attack targets. Spine/Spore
# crawlers are deliberately NOT here: a crawler creeping towards our base
# is a real threat.
ZERG_NON_COMBAT = frozenset({
    104,  # Drone
    106,  # Overlord
    150,  # Egg
    151,  # Larva
    893,  # Overlord cocoon
    86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,   # structures
    100, 101, 102, 137, 138, 142,                     # structures
})

# Terran SCV construction abilities. An SCV whose current order is one of
# these is welding a building and must NOT be pulled onto another job.
TERRAN_BUILD_ABILITIES = frozenset(range(318, 334))

# Footprint radius per structure, used for legal placement checks.
# (CC is 5x5, Barracks 3x3, Depot 2x2.)
FOOTPRINT_RADIUS = {
    CC: 2.5,
    ORBITAL: 2.5,
    PLANETARY: 2.5,
    BARRACKS: 1.5,
    BARRACKS_FLYING: 1.5,
    BUNKER: 1.5,
    DEPOT: 1.0,
    DEPOT_LOWERED: 1.0,
}

BUILD_RADIUS = {"depot": 1.0, "barracks": 1.5, "bunker": 1.5, "cc": 2.5}
PLACEMENT_MARGIN = 0.6
RESOURCE_MARGIN = 2.0

# ------------------------------------------------------------- difficulty
# A one-base, marine-only, no-upgrade build beats the Easy bot and loses
# to anything above it: harder built-in bots attack earlier, with more
# and tougher units, and out-produce a single mineral line. The profiles
# below change what the bot invests in, not just a few thresholds -
# harder opponents get an expansion, bunkers, a bigger standing guard
# and a later, larger attack timing.
PROFILES = {
    "easy": {
        "worker_target": 20,
        "max_barracks": 6,
        "attack_min": 15,
        "attack_min_napad": 12,
        "min_attackers": 8,
        "home_reserve": 4,
        "defense_radius": 20.0,
        "bunkers": 0,
        "expand_after_barracks": None,
        "scout_min_workers": 14,
    },
    "medium": {
        "worker_target": 26,
        "max_barracks": 7,
        "attack_min": 20,
        "attack_min_napad": 17,
        "min_attackers": 12,
        "home_reserve": 6,
        "defense_radius": 24.0,
        "bunkers": 1,
        "expand_after_barracks": 3,
        "scout_min_workers": 15,
    },
    "hard": {
        "worker_target": 32,
        "max_barracks": 8,
        "attack_min": 26,
        "attack_min_napad": 22,
        "min_attackers": 16,
        "home_reserve": 8,
        "defense_radius": 26.0,
        "bunkers": 2,
        "expand_after_barracks": 3,
        "scout_min_workers": 16,
    },
}

# Difficulties above "hard" reuse the hard profile.
PROFILE_FOR_DIFFICULTY = {
    "very_easy": "easy", "easy": "easy",
    "medium": "medium", "medium_hard": "medium",
    "hard": "hard", "harder": "hard", "very_hard": "hard",
    "cheat_vision": "hard", "cheat_money": "hard", "cheat_insane": "hard",
}


def profile_for(name):
    return dict(PROFILES.get(name, PROFILES["easy"]))


# Defaults, used when nothing else is specified.
WORKER_TARGET = PROFILES["easy"]["worker_target"]
MAX_BARRACKS = PROFILES["easy"]["max_barracks"]
ATTACK_ARMY_MIN = PROFILES["easy"]["attack_min"]
ATTACK_ARMY_MIN_NAPAD = PROFILES["easy"]["attack_min_napad"]
MIN_ATTACKERS = PROFILES["easy"]["min_attackers"]
HOME_RESERVE = PROFILES["easy"]["home_reserve"]
DEFENSE_RADIUS = PROFILES["easy"]["defense_radius"]
DEFENSE_RADIUS_SQ = DEFENSE_RADIUS ** 2
RALLY_DISTANCE = 14.0
GUARD_DISTANCE = 5.0

# Idle acceleration: when nothing needs doing and no enemy army is in
# sight, the environment is asked to advance more loops per step. SC2's
# per-observation overhead dominates wall-clock, so fewer, larger steps
# is by far the biggest speed lever available to an agent.
IDLE_STEP_FACTOR = 3
MAX_STEP_MUL = 24

# Cooldowns in game loops (step_mul 8 => one agent step is 8 loops).
CD = {
    "decision": 224,
    "scv": 32,
    "marine": 16,
    "depot": 64,
    "barracks": 80,
    "attack": 64,
    "defend": 32,
    "move": 160,
    "harvest": 48,
    "scout": 1200,
    "orbital": 200,
    "mule": 32,
    "lower": 160,
    "bunker": 120,
    "load": 96,
    "expand": 240,
    "expand_wanted": 1,
}

# How long a worker stays reserved for a build job.
BUILDER_BUSY_LOOPS = {"depot": 700, "barracks": 1300,
                      "bunker": 800, "cc": 1400}

# A build attempt that has not produced a structure by then is treated as
# a bad placement. Must be generous enough for the SCV to walk there.
BUILD_TIMEOUT = 240
# Builder is standing on the spot and nothing appeared -> illegal placement.
ARRIVED_TIMEOUT = 48
# An enemy we have been attacking for this long without it dying is
# almost certainly a stale snapshot, or unreachable.
STUCK_TARGET_LOOPS = 900
DEAD_TARGET_COOLDOWN = 4000
# A spot that failed once is probably terrain, so back off harder each
# time it fails - but never permanently, because a spot can free up when
# a building there is destroyed.
FAILED_POINT_COOLDOWN = 480
FAILED_POINT_COOLDOWN_MAX = 5000

# Every Terran main starts with its mineral line about this far from the
# Command Center, on every map. Comparing the observed distance against
# it tells the agent how many raw units one world tile is worth, which is
# what makes the placement geometry work outside Simple64.
MINERAL_LINE_DISTANCE = 6.4

RING_POINTS = 12
BARRACKS_RINGS = (5.0, 7.0, 9.0, 11.0, 13.0, 15.0)
DEPOT_RINGS = (4.5, 6.5, 8.5, 10.5, 12.5, 14.5)

VERBOSE = True


def log(message):
    if VERBOSE:
        print(message)


# ======================================================================
# HELPERS
# ======================================================================

def normalise_progress(value):
    """build_progress comes back as an integer percentage (0-100).

    pysc2 discretises it with ``int(u.build_progress * 100)`` because
    raw_units is an int array, so comparing it against 0.999 treats a
    structure that is 1% done as finished - which silently made
    ``barracks_building`` and ``depot_building`` permanently empty. The
    test below is written so the code is correct under either scale.
    """
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def dist2(ax, ay, bx, by):
    """Squared distance - avoids a sqrt in every proximity test."""
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


# ======================================================================
# OBSERVATION SNAPSHOT
# ======================================================================

# The only raw_unit fields this agent needs, in the order they are
# unpacked below. Slicing these 11 columns out before converting to
# Python is ~3x cheaper than converting all 46.
NEEDED_COLUMNS = ("unit_type", "alliance", "x", "y", "build_progress",
                  "order_length", "energy", "cargo_space_taken",
                  "display_type", "tag",
                  "order_id_0", "order_id_1", "order_id_2", "order_id_3")

# display_type in the raw observation: 1 = currently visible, 2 = a
# remembered snapshot of something we saw earlier and may well be gone.
DISPLAY_VISIBLE = 1


def _column_index():
    """Column indices of raw_units, read from the installed pysc2.

    Read from features.FeatureUnit rather than hard-coded, so a pysc2
    version with a different layout still works.
    """
    feature_unit = getattr(features, "FeatureUnit", None)
    if feature_unit is None:
        return None

    indices = []
    for name in NEEDED_COLUMNS:
        index = getattr(feature_unit, name, None)
        if index is None:
            return None
        indices.append(int(index))

    return np.array(indices, dtype=np.intp)


COLUMN_INDEX = _column_index()
MAX_COLUMN = int(COLUMN_INDEX.max()) if COLUMN_INDEX is not None else -1


class UnitView(object):
    """A single unit, read out of the observation exactly once."""

    __slots__ = ("tag", "unit_type", "alliance", "x", "y",
                 "progress", "orders", "energy", "cargo", "visible",
                 "order_ids")

    def __init__(self, tag, unit_type, alliance, x, y, progress, orders,
                 energy=0, cargo=0, visible=True, order_ids=()):
        self.tag = tag
        self.unit_type = unit_type
        self.alliance = alliance
        self.x = x
        self.y = y
        self.progress = progress
        self.orders = orders
        self.energy = energy
        self.cargo = cargo
        self.visible = visible
        self.order_ids = order_ids


class Snapshot(object):
    """One pass over raw_units, indexed for O(1) lookups afterwards."""

    __slots__ = ("scv_by_tag", "own_by_type", "enemy", "resources",
                 "own_buildings")

    def __init__(self, raw_units):
        # Only SCVs are ever looked up by tag (is my builder still alive,
        # is my scout still alive), so the other ~120 units per step skip
        # the dict insert entirely.
        self.scv_by_tag = {}
        self.own_by_type = {}
        self.enemy = []
        self.resources = []
        self.own_buildings = []

        rows = self._rows(raw_units)

        if rows is not None:
            self._read_rows(rows)
        else:
            self._read_objects(raw_units)

    # -------------------------------------------------------------- input

    @staticmethod
    def _rows(raw_units):
        """Whole observation converted to plain lists in one C-level call.

        pysc2 hands back a NamedNumpyArray; touching it field by field is
        the single most expensive thing a raw agent can do.
        """
        if COLUMN_INDEX is None or raw_units is None:
            return None

        if not isinstance(raw_units, np.ndarray):
            return None

        # asarray() strips the NamedNumpyArray wrapper, so the slice below
        # is plain numpy indexing instead of pysc2's name resolution.
        base = np.asarray(raw_units)

        if base.ndim != 2 or base.shape[0] == 0:
            return []

        if base.shape[1] <= MAX_COLUMN:
            return None

        return base[:, COLUMN_INDEX].tolist()

    def _read_rows(self, rows):
        # Runs over every unit on the map on every step, so the bucketing
        # is inlined here instead of going through _index() per unit.
        scv_by_tag = self.scv_by_tag
        own_by_type = self.own_by_type
        enemy = self.enemy
        resources = self.resources
        own_buildings = self.own_buildings

        for row in rows:
            (unit_type, alliance, x, y, progress,
             orders, energy, cargo, display, tag, o0, o1, o2, o3) = row

            if alliance == ALLIANCE_ENEMY:
                enemy.append(UnitView(
                    tag, unit_type, alliance, x, y,
                    progress * 0.01 if progress > 1 else float(progress), 0,
                    0, 0, display == DISPLAY_VISIBLE))
                continue

            if alliance == ALLIANCE_NEUTRAL:
                if unit_type in RESOURCE_IDS:
                    resources.append(
                        UnitView(tag, unit_type, alliance, x, y, 1.0, 0))
                continue

            if alliance != ALLIANCE_SELF:
                continue

            if unit_type == SCV:
                view = UnitView(tag, unit_type, alliance, x, y, 1.0,
                                orders, 0, 0, True, (o0, o1, o2, o3))
                scv_by_tag[tag] = view
            else:
                view = UnitView(
                    tag, unit_type, alliance, x, y,
                    progress * 0.01 if progress > 1 else float(progress),
                    orders, energy, cargo)

            bucket = own_by_type.get(unit_type)
            if bucket is None:
                own_by_type[unit_type] = [view]
            else:
                bucket.append(view)

            radius = FOOTPRINT_RADIUS.get(unit_type)
            if radius is not None:
                own_buildings.append((x, y, radius))

    def _read_objects(self, raw_units):
        """Fallback for observations that are not a plain raw ndarray.

        Works on anything exposing the fields as attributes (pysc2's
        NamedNumpyArray, or plain objects in a test harness). If the rows
        expose nothing usable, the step degrades to "no units seen"
        instead of crashing the episode.
        """
        try:
            self._read_objects_inner(raw_units)
        except AttributeError as e:
            log("[OBS] unreadable raw_units (%s) - treating as empty" % e)

    def _read_objects_inner(self, raw_units):
        for u in raw_units if raw_units is not None else ():
            alliance = int(u.alliance)
            unit_type = int(u.unit_type)

            if alliance == ALLIANCE_NEUTRAL and unit_type not in RESOURCE_IDS:
                continue
            if alliance not in (ALLIANCE_SELF, ALLIANCE_NEUTRAL, ALLIANCE_ENEMY):
                continue

            orders = getattr(u, "order_length", None)
            if orders is None:
                queued = getattr(u, "orders", None)
                orders = len(queued) if queued is not None else 0

            order_ids = ()
            if alliance == ALLIANCE_SELF and unit_type == SCV:
                order_ids = tuple(
                    int(getattr(u, "order_id_%d" % i, 0) or 0) for i in range(4))

            self._index(UnitView(
                int(u.tag), unit_type, alliance,
                float(u.x), float(u.y),
                normalise_progress(getattr(u, "build_progress", 1.0)),
                int(orders), int(getattr(u, "energy", 0) or 0),
                int(getattr(u, "cargo_space_taken", 0) or 0),
                int(getattr(u, "display_type", 1) or 1) == DISPLAY_VISIBLE,
                order_ids,
            ))

    # ------------------------------------------------------------- index

    def _index(self, view):
        if view.alliance == ALLIANCE_SELF:
            if view.unit_type == SCV:
                self.scv_by_tag[view.tag] = view

            bucket = self.own_by_type.get(view.unit_type)
            if bucket is None:
                self.own_by_type[view.unit_type] = [view]
            else:
                bucket.append(view)

            radius = FOOTPRINT_RADIUS.get(view.unit_type)
            if radius is not None:
                self.own_buildings.append((view.x, view.y, radius))

        elif view.alliance == ALLIANCE_ENEMY:
            self.enemy.append(view)

        else:
            self.resources.append(view)

    # ------------------------------------------------------------ queries

    EMPTY = ()

    def of_type(self, *unit_types):
        buckets = self.own_by_type

        if len(unit_types) == 1:
            return buckets.get(unit_types[0], Snapshot.EMPTY)

        found = [buckets[t] for t in unit_types if t in buckets]

        if not found:
            return Snapshot.EMPTY
        if len(found) == 1:
            return found[0]          # no copy in the common case

        out = []
        for bucket in found:
            out.extend(bucket)
        return out


def complete(unit_list):
    return [u for u in unit_list if u.progress >= 0.999]


def incomplete(unit_list):
    return [u for u in unit_list if u.progress < 0.999]


# ======================================================================
# MCDM
# ======================================================================
#
# The decision problem is a proper matrix: 7 alternatives x 6 criteria.
# This matters - with a single scalar utility per alternative, SAW,
# TOPSIS and VIKOR are mathematically guaranteed to produce the same
# ranking, so any "comparison" of them would be vacuous. Criteria are
# kept in their natural units (minerals, game loops, ratios) and are
# normalised inside each method, so the normalisation scheme is an
# explicit experimental variable rather than something baked into the
# matrix.

CRITERIA = ("ekonomski_ucinak", "vojni_ucinak", "hitnost",
            "trosak_min", "rizik", "latencija")

# +1 = benefit (more is better), -1 = cost (less is better)
CRITERION_TYPE = (1, 1, 1, -1, -1, -1)

CRITERION_WEIGHT = (0.20, 0.25, 0.30, 0.10, 0.10, 0.05)

ACTIONS = (
    "EKONOMIJA",
    "PROIZVODNJA",
    "NAPAD",
    "OBRANA",
    "IZVIDANJE",
    "POVLACENJE",
    "ODRZAVANJE",
)


# ----------------------------------------------------------------------
# NORMALISATION
# ----------------------------------------------------------------------

def normalise_linear(matrix, types):
    """Max-min normalisation, returned benefit-oriented in [0, 1].

    A column where every alternative scores the same carries no
    information, so it is set to 1.0 for all of them rather than
    dividing by zero.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    out = [[0.0] * cols for _ in range(rows)]

    for j in range(cols):
        column = [matrix[i][j] for i in range(rows)]
        hi = max(column)
        lo = min(column)
        span = hi - lo

        for i in range(rows):
            if span <= 1e-12:
                out[i][j] = 1.0
            elif types[j] > 0:
                out[i][j] = (column[i] - lo) / span
            else:
                out[i][j] = (hi - column[i]) / span

    return out


def normalise_vector(matrix, types):
    """Euclidean (vector) normalisation - the classic TOPSIS scheme.

    Orientation is preserved: cost columns stay cost columns, and the
    method handles them when it picks the ideal point.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    out = [[0.0] * cols for _ in range(rows)]

    for j in range(cols):
        norm = math.sqrt(sum(matrix[i][j] ** 2 for i in range(rows)))

        for i in range(rows):
            out[i][j] = 0.0 if norm <= 1e-12 else matrix[i][j] / norm

    return out


# ----------------------------------------------------------------------
# AGGREGATION METHODS
# ----------------------------------------------------------------------

class MethodResult(object):
    """Scores and ranking produced by one aggregation method."""

    __slots__ = ("scores", "ranking", "higher_better", "meta")

    def __init__(self, scores, higher_better, meta=None):
        self.scores = scores
        self.higher_better = higher_better
        self.meta = meta or {}
        self.ranking = sorted(
            scores, key=lambda i: (-scores[i] if higher_better else scores[i], i))


def submatrix(matrix, feasible):
    return [matrix[i] for i in feasible]


def saw(matrix, weights, types, feasible, scheme="linear", **kwargs):
    """Simple Additive Weighting. Always max-min normalised - that is
    the standard SAW definition and keeps cost criteria meaningful."""
    sub = submatrix(matrix, feasible)
    norm = normalise_linear(sub, types)

    scores = {}
    for k, i in enumerate(feasible):
        scores[i] = sum(weights[j] * norm[k][j] for j in range(len(weights)))

    return MethodResult(scores, higher_better=True)


def topsis(matrix, weights, types, feasible, scheme="vector", **kwargs):
    """Technique for Order Preference by Similarity to Ideal Solution."""
    sub = submatrix(matrix, feasible)

    if scheme == "vector":
        norm = normalise_vector(sub, types)
        orientation = types
    else:
        norm = normalise_linear(sub, types)
        orientation = (1,) * len(types)

    cols = len(weights)
    weighted = [[weights[j] * row[j] for j in range(cols)] for row in norm]

    best = []
    worst = []
    for j in range(cols):
        column = [row[j] for row in weighted]
        if orientation[j] > 0:
            best.append(max(column))
            worst.append(min(column))
        else:
            best.append(min(column))
            worst.append(max(column))

    scores = {}
    separations = {}

    for k, i in enumerate(feasible):
        d_best = math.sqrt(
            sum((weighted[k][j] - best[j]) ** 2 for j in range(cols)))
        d_worst = math.sqrt(
            sum((weighted[k][j] - worst[j]) ** 2 for j in range(cols)))

        total = d_best + d_worst
        # Every alternative identical -> all equally close to the ideal.
        scores[i] = 1.0 if total <= 1e-12 else d_worst / total
        separations[i] = (d_best, d_worst)

    return MethodResult(scores, higher_better=True,
                        meta={"separations": separations})


def vikor(matrix, weights, types, feasible, v=0.5, **kwargs):
    """VIKOR compromise ranking.

    Returns Q (lower is better) plus the two acceptance conditions.
    Unlike TOPSIS, VIKOR does not always propose a single winner: when
    C1 fails the result is a compromise SET, and an agent still has to
    break the tie somehow. How often that happens is one of the more
    interesting things to report.
    """
    sub = submatrix(matrix, feasible)
    cols = len(weights)
    count = len(feasible)

    best = []
    worst = []
    for j in range(cols):
        column = [row[j] for row in sub]
        if types[j] > 0:
            best.append(max(column))
            worst.append(min(column))
        else:
            best.append(min(column))
            worst.append(max(column))

    s_values = {}
    r_values = {}

    for k, i in enumerate(feasible):
        terms = []
        for j in range(cols):
            span = best[j] - worst[j]
            # Non-discriminating criterion contributes nothing.
            gap = 0.0 if abs(span) <= 1e-12 else (best[j] - sub[k][j]) / span
            terms.append(weights[j] * gap)

        s_values[i] = sum(terms)
        r_values[i] = max(terms)

    s_best, s_worst = min(s_values.values()), max(s_values.values())
    r_best, r_worst = min(r_values.values()), max(r_values.values())

    s_span = s_worst - s_best
    r_span = r_worst - r_best

    q_values = {}
    for i in feasible:
        s_part = 0.0 if s_span <= 1e-12 else (s_values[i] - s_best) / s_span
        r_part = 0.0 if r_span <= 1e-12 else (r_values[i] - r_best) / r_span
        q_values[i] = v * s_part + (1.0 - v) * r_part

    result = MethodResult(q_values, higher_better=False)
    ranking = result.ranking

    first = ranking[0]
    second = ranking[1] if count > 1 else None

    # C1 - acceptable advantage.
    threshold = 1.0 / (count - 1) if count > 1 else 0.0
    c1 = True if second is None else (
        q_values[second] - q_values[first] >= threshold)

    # C2 - acceptable stability in decision making.
    best_s = min(s_values, key=lambda i: (s_values[i], i))
    best_r = min(r_values, key=lambda i: (r_values[i], i))
    c2 = first in (best_s, best_r)

    # The compromise set.
    if not c1:
        compromise = [first]
        for candidate in ranking[1:]:
            if q_values[candidate] - q_values[first] < threshold:
                compromise.append(candidate)
            else:
                break
    elif not c2:
        compromise = [first, second] if second is not None else [first]
    else:
        compromise = [first]

    result.meta = {
        "S": s_values,
        "R": r_values,
        "c1": c1,
        "c2": c2,
        "compromise": compromise,
        "v": v,
    }

    return result


METHODS = {
    "saw": saw,
    "topsis": topsis,
    "vikor": vikor,
}

DEFAULT_SCHEME = {
    "saw": "linear",
    "topsis": "vector",
    "vikor": "linear",
}


# ----------------------------------------------------------------------
# DECISION
# ----------------------------------------------------------------------

class Decision(object):
    __slots__ = ("action", "index", "matrix", "weights", "available",
                 "result", "method", "scheme_used", "v_used")

    def __init__(self, action, index, matrix, weights, available, result,
                 method, scheme_used="", v_used=0.0):
        self.action = action
        self.index = index
        self.matrix = matrix
        self.weights = weights
        self.available = available
        self.result = result
        self.method = method
        self.scheme_used = scheme_used
        self.v_used = v_used


class MCDM:
    """Builds the decision matrix and ranks the alternatives.

    The aggregation method is swappable so that SAW, TOPSIS and VIKOR
    can be compared on the identical decision problem.
    """

    ACTIONS = list(ACTIONS)

    def __init__(self, method="saw", scheme=None, v=0.5,
                 weights=CRITERION_WEIGHT, cfg=None):
        self.cfg = cfg or profile_for("easy")
        self.method_name = method
        self.method = METHODS.get(method, saw)
        self.scheme = scheme or DEFAULT_SCHEME.get(method, "linear")
        self.v = v
        self.weights = tuple(weights)

    # ------------------------------------------------------------------
    # THE DECISION MATRIX
    # ------------------------------------------------------------------

    def decision_matrix(self, s):
        """One row per alternative, in the natural units of each criterion.

        Columns: economic effect, military effect, urgency (all benefit,
        0..1), mineral cost (minerals), risk (0..1), latency (game loops
        until the action pays off).
        """
        workers = s["workers"]
        army = s["army"]
        enemy = s["enemy"]
        minerals = s["minerals"]
        barracks = s["barracks"]
        threats = len(s["threats_home"])
        supply = s["supply_left"] + s["supply_pending"]

        cfg = self.cfg
        worker_target = float(cfg["worker_target"])
        max_barracks = float(cfg["max_barracks"])
        attack_min = float(cfg["attack_min"])

        worker_deficit = clamp((worker_target - workers) / worker_target)
        barracks_deficit = clamp((max_barracks - barracks) / max_barracks)
        mineral_surplus = clamp(minerals / 400.0)
        force_ratio = (army + 1.0) / (enemy + 1.0)
        supply_pressure = clamp(1.0 - supply / 8.0)

        can_expand = barracks + len(s["barracks_building"]) < max_barracks

        return [
            # EKONOMIJA - train an SCV / put idle workers back on minerals
            [worker_deficit,
             0.05,
             worker_deficit * (1.0 if minerals >= SCV_COST else 0.4),
             float(SCV_COST),
             0.10,
             300.0],

            # PROIZVODNJA - another barracks, or marines out of the ones we have
            [0.05,
             clamp(0.6 * barracks_deficit + 0.4 * mineral_surplus),
             mineral_surplus,
             float(BARRACKS_COST if can_expand else MARINE_COST),
             0.20,
             1030.0 if can_expand else 400.0],

            # NAPAD
            [0.0,
             clamp(army / attack_min) * clamp(force_ratio / 1.5),
             clamp(army / attack_min),
             0.0,
             clamp(1.0 - force_ratio / 2.0),
             200.0],

            # OBRANA
            [0.15,
             clamp(threats / max(1.0, army * 0.5)) if threats else 0.0,
             1.0 if threats else 0.0,
             0.0,
             clamp(threats / max(1.0, float(army))) * 0.5,
             0.0],

            # IZVIDANJE
            [0.0,
             0.10,
             0.30 if s["enemy_seen"] == 0 else 0.05,
             0.0,
             0.30,
             600.0],

            # POVLACENJE
            [0.05,
             clamp((enemy - army) / max(1.0, float(enemy))) if army else 0.0,
             clamp((enemy - army) / max(1.0, float(enemy))) if army else 0.0,
             0.0,
             0.20,
             150.0],

            # ODRZAVANJE - supply, idle workers, rally, MULE
            [0.20,
             0.10,
             clamp(0.6 * supply_pressure + 0.4 * mineral_surplus),
             float(DEPOT_COST),
             0.05,
             470.0],
        ]

    # ------------------------------------------------------------------
    # FEASIBILITY
    # ------------------------------------------------------------------

    def available(self, s):
        cfg = self.cfg
        army = s["army"]
        threats = len(s["threats_home"])

        can_expand = (s["barracks"] + len(s["barracks_building"])
                      < cfg["max_barracks"])
        can_train = s["barracks"] > 0

        return {
            "EKONOMIJA": s["workers"] < cfg["worker_target"],
            "PROIZVODNJA": can_expand or can_train,
            "NAPAD": army >= cfg["attack_min_napad"],
            "OBRANA": threats > 0,
            "IZVIDANJE": (s["enemy_seen"] == 0
                          and s["workers"] >= cfg["scout_min_workers"]),
            "POVLACENJE": army > 0 and threats == 0,
            "ODRZAVANJE": True,
        }

    # ------------------------------------------------------------------

    def decide(self, s):
        matrix = self.decision_matrix(s)
        available = self.available(s)

        feasible = [i for i, name in enumerate(ACTIONS) if available[name]]

        if not feasible:
            feasible = [ACTIONS.index("ODRZAVANJE")]

        result = self.method(matrix, self.weights, CRITERION_TYPE, feasible,
                             scheme=self.scheme, v=self.v)

        index = result.ranking[0]

        return Decision(ACTIONS[index], index, matrix, self.weights,
                        available, result, self.method_name,
                        self.scheme, self.v)


# ======================================================================
# MAP RESOLUTION
# ======================================================================
#
# pysc2 does not look in the Maps folder: maps.get() searches a registry
# built from lib.Map subclasses, so a .SC2Map file that is present on
# disk but not registered raises NoMapError. Ladder maps are also
# registered under a different name than their file ("Acropolis" vs
# "AcropolisLE"). Both cases are handled here.

def _normalise_map_name(name):
    plain = "".join(c for c in str(name).lower() if c.isalnum())
    return plain[:-2] if plain.endswith("le") else plain


def find_map_file(name):
    """Look for <name>.SC2Map inside the StarCraft II Maps folder."""
    try:
        from pysc2 import run_configs
        data_dir = run_configs.get().data_dir
    except Exception as e:  # noqa: BLE001
        print("[MAP] cannot locate the SC2 install (%s)" % e)
        return None, None

    root = os.path.join(data_dir, "Maps")
    target = _normalise_map_name(name)

    if not os.path.isdir(root):
        return None, None

    for directory, _, files in os.walk(root):
        for filename in files:
            if not filename.endswith(".SC2Map"):
                continue
            if _normalise_map_name(filename[:-7]) != target:
                continue

            relative = os.path.relpath(directory, root)
            return filename[:-7], ("" if relative == "." else relative)

    return None, None


def resolve_map(name, directory=""):
    """Return a map name pysc2 will accept, registering it if needed."""
    from pysc2 import maps
    from pysc2.maps import lib as maps_lib

    try:
        maps.get(name)
        return name
    except Exception:  # noqa: BLE001 - NoMapError and friends
        pass

    registry = maps.get_maps()
    target = _normalise_map_name(name)

    # Registered under a slightly different name, e.g. Acropolis / AcropolisLE.
    for key, map_class in registry.items():
        names = [key, getattr(map_class, "filename", "") or ""]
        if any(_normalise_map_name(n) == target for n in names if n):
            print("[MAP] '%s' -> registered as '%s'" % (name, key))
            return key

    # Present on disk but never registered: register it now.
    filename, found_directory = find_map_file(name)

    if filename:
        directory = directory or found_directory
        print("[MAP] '%s' found in Maps/%s - registering it"
              % (filename, directory or "."))

        type(str(name), (maps_lib.Map,), {
            "filename": filename,
            "directory": directory,
            "players": 2,
            "game_steps_per_episode": 0,
            "download": "",
        })

        return name

    similar = sorted(k for k in registry if target[:5] in _normalise_map_name(k))

    raise ValueError(
        "Map %r is not registered with pysc2 and no matching .SC2Map was "
        "found in the Maps folder.\n"
        "pysc2 knows %d maps; similar names: %s\n"
        "Put the file in StarCraft II/Maps (a subfolder is fine) or pass "
        "--map_dir with the subfolder name."
        % (name, len(registry), ", ".join(similar[:10]) or "none"))


# ======================================================================
# DECISION LOG
# ======================================================================

class DecisionLog(object):
    """Long-format CSV: one row per alternative per decision.

    Keeping the raw matrix (not just the winner) is what makes the
    offline comparison possible - every logged decision can be re-ranked
    by any method afterwards, on identical inputs.
    """

    COLUMNS = (["episode", "loop", "method", "scheme", "v", "action",
                "feasible"]
               + list(CRITERIA)
               + ["weight_sum", "score", "rank", "chosen",
                  "workers", "army", "barracks", "minerals", "supply_left",
                  "enemy", "threats"])

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.COLUMNS)

    def write(self, episode, loop, decision, s):
        result = decision.result
        rank_of = {index: rank for rank, index in enumerate(result.ranking, 1)}

        context = [s["workers"], s["army"], s["barracks"], s["minerals"],
                   s["supply_left"], s["enemy"], len(s["threats_home"])]

        for i, name in enumerate(ACTIONS):
            feasible = decision.available[name]

            self.writer.writerow(
                [episode, loop, decision.method, decision.scheme_used,
                 decision.v_used, name, int(feasible)]
                + ["%.6g" % value for value in decision.matrix[i]]
                + ["%.6g" % sum(decision.weights),
                   "%.6g" % result.scores[i] if i in result.scores else "",
                   rank_of.get(i, ""),
                   int(i == decision.index)]
                + context)

    def close(self):
        try:
            self.file.close()
        except Exception:  # noqa: BLE001
            pass


# ======================================================================
# AGENT
# ======================================================================

class WinAgent:

    def __init__(self, step_mul=8, idle_speedup=True, method="saw",
                 scheme=None, vikor_v=0.5, decision_log=None,
                 profile="easy", raw_resolution=RAW_RESOLUTION):
        self.profile_name = profile
        self.cfg = profile_for(profile)
        self.raw_resolution = float(raw_resolution)
        # method="none" is the ablation baseline: no strategic layer at
        # all, only the hard constraints and the fallback chain. Without
        # it there is no way to claim MCDM contributes anything.
        self.method_name = method
        self.mcdm = None if method == "none" else MCDM(
            method=method, scheme=scheme, v=vikor_v, cfg=self.cfg)

        self.decision_log = decision_log
        self.episode = 0
        self.orders = Counter()
        self.base_step_mul = step_mul

        # Kept on the instance rather than read from FLAGS: this is on the
        # hot path, and touching an unparsed flag raises - which would
        # break anyone importing this agent from their own runner.
        self.idle_speedup = idle_speedup

        self.reset()

    # ------------------------------------------------------------------
    # EPISODE STATE
    # ------------------------------------------------------------------

    def reset(self):
        self.current_loop = 0
        self.last = {}
        self.home = None
        self.decisions = 0
        # With no strategic layer the dispatch table is simply empty.
        self.action = None if self.method_name == "none" else "ODRZAVANJE"

        # Workers reserved for a job: tag -> loop until which it is busy.
        self.busy_workers = {}
        self.scout_tag = None
        self.rallied = set()

        # How many game loops the runner should advance after this step.
        self.next_step_mul = self.base_step_mul

        self.failed_points = {"depot": {}, "barracks": {},
                              "bunker": {}, "cc": {}}

        # One outstanding build attempt per kind. PySC2's raw interface has
        # no "can I build here" query, so every placement is a guess that is
        # only validated by trying it; issuing several at once makes it
        # impossible to tell which guess was the bad one.
        self.build_attempts = {"depot": None, "barracks": None,
                               "bunker": None, "cc": None}

        # Points where a build actually succeeded. Ground next to proven
        # ground is very likely buildable too (same plateau), so it is
        # searched before guessing further out into possible cliffs/water.
        self.verified_ground = []

        # World scale, measured from the mineral line on the first
        # observation. 1.0 means one raw unit is one world tile.
        self.calibrated = False
        self.expansion_wanted = False
        self.attack_target = None
        self.attack_target_since = 0
        self.dead_targets = {}
        self.apply_scale(1.0)

        # Caches. Ring points depend only on our start location, and the
        # mineral line does not move, so neither is recomputed per step.
        self._ring_cache = {}
        self._patch_cache = None
        self._patch_cache_size = 0
        self._patch_cache_loop = -99999

    # ------------------------------------------------------------------
    # WORLD SCALE
    # ------------------------------------------------------------------

    def apply_scale(self, scale):
        """Re-express every distance in the observation's own units."""
        self.scale = scale
        cfg = self.cfg

        self.footprint = {k: v * scale for k, v in FOOTPRINT_RADIUS.items()}
        self.build_radius = {k: v * scale for k, v in BUILD_RADIUS.items()}
        self.placement_margin = PLACEMENT_MARGIN * scale
        self.resource_margin = RESOURCE_MARGIN * scale

        self.rings = {
            "barracks": [r * scale for r in BARRACKS_RINGS],
            "depot": [r * scale for r in DEPOT_RINGS],
        }

        self.defense_radius_sq = (cfg["defense_radius"] * scale) ** 2
        self.rally_distance = RALLY_DISTANCE * scale
        self.guard_distance = GUARD_DISTANCE * scale

        self.patch_radius_sq = (15.0 * scale) ** 2
        self.cluster_radius_sq = (12.0 * scale) ** 2
        self.expansion_min_sq = (10.0 * scale) ** 2
        self.arrived_radius_sq = (2.5 * scale) ** 2

        self.build_tolerance_sq = {
            "depot": (2.0 * scale) ** 2,
            "barracks": (2.5 * scale) ** 2,
            "bunker": (2.5 * scale) ** 2,
            "cc": (3.0 * scale) ** 2,
        }

        self.edge = max(2.0, 3.0 * scale)

        if hasattr(self, "_ring_cache"):
            self._ring_cache = {}

    def calibrate(self, snap):
        """Measure how many raw units a world tile is worth.

        pysc2 squeezes the whole map into raw_resolution, so on a 176
        tile ladder map one raw unit is ~2.7 world tiles. Without this
        every footprint, ring and radius in the agent would be off by
        that factor - which is why a build that works on Simple64 falls
        apart on AcropolisLE.
        """
        if self.calibrated:
            return

        home = self.get_home(snap)

        if home is None or len(snap.resources) < 4:
            return

        hx, hy = home
        near = sorted(dist2(hx, hy, r.x, r.y) for r in snap.resources)[:8]

        if len(near) < 4:
            return

        median = math.sqrt(near[len(near) // 2])

        if median <= 0.5:
            return

        scale = clamp(median / MINERAL_LINE_DISTANCE, 0.2, 5.0)

        self.calibrated = True
        self.apply_scale(scale)

        log("[SCALE] mineral line at %.1f -> 1 world tile = %.2f raw units"
            % (median, scale))

    # ------------------------------------------------------------------
    # COOLDOWNS
    # ------------------------------------------------------------------

    def ready(self, key, loop):
        return loop - self.last.get(key, -999999) >= CD[key]

    def mark(self, key, loop):
        self.last[key] = loop

    def since(self, key, loop):
        return loop - self.last.get(key, -999999)

    # ------------------------------------------------------------------
    # OBSERVATION
    # ------------------------------------------------------------------

    def queue(self, unit):
        """Number of queued orders.

        FIX: the raw observation exposes ``order_length``; there is no
        ``orders`` field. The original code read a non-existent attribute
        through ``getattr(..., None)``, so this always returned 0 and the
        whole "zero queue" policy silently did nothing. The snapshot now
        resolves this once per unit per step.
        """
        return unit.orders

    def is_constructing(self, unit):
        """True if this SCV is currently welding a building."""
        for ability in unit.order_ids:
            if ability in TERRAN_BUILD_ABILITIES:
                return True
        return False

    def get_home(self, snap):
        """Cache our start location the first time we see a base.

        Simple64 has two mirrored spawns, so every build point and rally
        point is derived from this instead of hard-coded coordinates.
        """
        if self.home is not None:
            return self.home

        bases = snap.of_type(*BASE_IDS)
        if bases:
            self.home = (bases[0].x, bases[0].y)

        return self.home

    def enemy_point(self, snap):
        """Mirror of our start location - the likely enemy main."""
        home = self.get_home(snap) or (self.raw_resolution / 2.0,
                                       self.raw_resolution / 2.0)
        centre = self.raw_resolution / 2.0

        edge = self.edge
        limit = self.raw_resolution - edge

        return (clamp(2.0 * centre - home[0], edge, limit),
                clamp(2.0 * centre - home[1], edge, limit))

    def staging_point(self, snap, distance=None):
        """A point `distance` units from home, towards the map centre."""
        if distance is None:
            distance = self.rally_distance
        home = self.get_home(snap) or (self.raw_resolution / 2.0,
                                       self.raw_resolution / 2.0)
        centre = self.raw_resolution / 2.0

        dx = centre - home[0]
        dy = centre - home[1]
        norm = max(1.0, math.hypot(dx, dy))

        edge = self.edge
        limit = self.raw_resolution - edge

        return (clamp(home[0] + dx / norm * distance, edge, limit),
                clamp(home[1] + dy / norm * distance, edge, limit))

    def guard_point(self, snap):
        """Just inside our own base - where new marines should wait.

        FIX: barracks used to be rallied to the forward staging point.
        Under pressure that walks every freshly built marine out of the
        base alone, straight into whatever is attacking, one at a time.
        A defender that trickles is a defender that loses.
        """
        return self.staging_point(snap, self.guard_distance)

    def rally_point(self, snap):
        return self.staging_point(snap, self.rally_distance)

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def state(self, snap, player):
        # Units (unlike structures) are never reported before they finish
        # training, so the buckets can be used directly - filtering them
        # would copy ~200 elements every step for nothing.
        scvs = snap.of_type(SCV)
        marines = snap.of_type(MARINE)

        barracks_all = snap.of_type(*BARRACKS_IDS)

        barracks_done = 0
        barracks_building = []
        for b in snap.of_type(BARRACKS):
            if b.progress >= 0.999:
                barracks_done += 1
            else:
                barracks_building.append(b)

        depots = snap.of_type(*DEPOT_IDS)
        depot_building = incomplete(depots)

        bases = complete(snap.of_type(*BASE_IDS))

        # Combat pressure only - drones/overlords/larvae are still valid
        # targets but must not make the bot think it is under attack.
        enemy_all = snap.enemy
        enemy_army = [
            u for u in enemy_all
            if u.unit_type not in ZERG_NON_COMBAT and u.progress >= 0.999
        ]

        home = self.get_home(snap)

        threats_home = []
        if home is not None and enemy_army:
            hx, hy = home
            radius_sq = self.defense_radius_sq
            threats_home = [
                u for u in enemy_army
                if dist2(hx, hy, u.x, u.y) < radius_sq
            ]

        food_used = int(player.food_used)
        food_cap = int(player.food_cap)

        return {
            "minerals": int(player.minerals),
            "food_used": food_used,
            "food_cap": food_cap,
            "supply_left": food_cap - food_used,
            "supply_pending": 8 * len(depot_building),

            "scvs": scvs,
            "workers": len(scvs),

            "marines": marines,
            "army": len(marines),

            "barracks": barracks_done,
            "barracks_all": barracks_all,
            "barracks_building": barracks_building,

            "depots": depots,
            "depot_building": depot_building,

            "cc": bases,
            "cc_building": incomplete(snap.of_type(*BASE_IDS)),
            "orbitals": len(snap.of_type(ORBITAL)),

            "enemy_all": enemy_all,
            "enemy_army": enemy_army,
            "enemy": len(enemy_army),
            "enemy_seen": len(enemy_all),
            "threats_home": threats_home,

            "home": home,
        }

    # ------------------------------------------------------------------
    # RAW ACTIONS
    # ------------------------------------------------------------------

    def call(self, name, *args):
        """Issue a raw function, returning None if it is not usable.

        FIX: the original gated every action on ``obs.available_actions``.
        That list is built for the *feature* action space and its ids do
        not correspond to RAW_FUNCTIONS ids, so the check compared two
        different id namespaces and could silently reject legal raw
        actions. The environment is created with
        ``ensure_available_actions=False``, so raw actions can be issued
        directly.
        """
        fn = getattr(actions.RAW_FUNCTIONS, name, None)

        if fn is None:
            log("[RAW MISSING] %s" % name)
            return None

        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001 - never crash the episode
            log("[RAW ERROR] %s: %s" % (name, e))
            return None

    def call_first(self, names, *args):
        for name in names:
            act = self.call(name, *args)
            if act is not None:
                return act
        return None

    def no_op(self):
        return actions.RAW_FUNCTIONS.no_op()

    # ------------------------------------------------------------------
    # WORKER BOOKKEEPING
    # ------------------------------------------------------------------

    def prune_busy(self, loop):
        if not self.busy_workers:
            return

        expired = [t for t, until in self.busy_workers.items() if until <= loop]
        for tag in expired:
            del self.busy_workers[tag]

    def free_workers(self, s):
        """SCVs that may be pulled onto a new job."""
        busy = self.busy_workers
        loop = self.current_loop
        scout = self.scout_tag

        out = []
        for u in s["scvs"]:
            tag = u.tag

            if tag == scout or busy.get(tag, -1) > loop:
                continue
            if u.order_ids and self.is_constructing(u):
                continue

            out.append(u)

        return out

    def pick_worker(self, s, point):
        """Closest free SCV to a point.

        FIX: the original required ``queue(u) == 0``. Every mining SCV has
        an order, so once the queue check is repaired that filter would
        match nothing and the bot would never build again. Pulling a
        mining worker is normal RTS play; what must be avoided is pulling
        one that is mid-construction, which is handled by free_workers().
        """
        candidates = self.free_workers(s)

        if not candidates:
            return None

        px, py = point
        return min(candidates, key=lambda u: dist2(px, py, u.x, u.y))

    # ------------------------------------------------------------------
    # BUILDING PLACEMENT
    # ------------------------------------------------------------------

    @staticmethod
    def key(point):
        return (round(point[0], 1), round(point[1], 1))

    def candidate_points(self, snap, kind):
        """Concentric rings around our real base, proven ground first.

        A small fixed list of points runs out as the base fills up, and
        then nothing can be built at all - depots included, which leaves
        supply stuck for the rest of the game. Rings at increasing radius
        always have somewhere further out to go.

        The rings depend only on our start location, so they are computed
        once per episode instead of re-running 72 sin/cos pairs on every
        placement search.
        """
        home = self.get_home(snap)

        if home is None:
            home = (self.raw_resolution / 2.0, self.raw_resolution / 2.0)

        cache_key = (kind, home)
        rings = self._ring_cache.get(cache_key)

        if rings is None:
            radii = self.rings["barracks" if kind == "barracks" else "depot"]
            offset = 0.0 if kind == "barracks" else math.pi / RING_POINTS

            rings = []
            for radius in radii:
                for i in range(RING_POINTS):
                    angle = offset + (2.0 * math.pi * i / RING_POINTS)
                    rings.append((
                        clamp(home[0] + radius * math.cos(angle),
                              self.edge, self.raw_resolution - self.edge),
                        clamp(home[1] + radius * math.sin(angle),
                              self.edge, self.raw_resolution - self.edge),
                    ))

            self._ring_cache[cache_key] = rings

        if not self.verified_ground:
            return rings

        points = []
        for vx, vy in self.verified_ground:
            for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4),
                           (3, 3), (-3, 3), (3, -3), (-3, -3)):
                points.append((
                    clamp(vx + dx * self.scale,
                          self.edge, self.raw_resolution - self.edge),
                    clamp(vy + dy * self.scale,
                          self.edge, self.raw_resolution - self.edge),
                ))

        points.extend(rings)
        return points

    def obstacles(self, snap, kind, resource_margin=None):
        """Blocked circles as (x, y, radius^2), built once per search.

        The old code re-scanned every unit in the observation for every
        candidate point - up to ~120 points x every unit on the map.
        """
        own_r = self.build_radius[kind]
        out = []

        for bx, by, br in snap.own_buildings:
            r = own_r + br * self.scale + self.placement_margin
            out.append((bx, by, r * r))

        r = own_r + (self.resource_margin if resource_margin is None
                     else resource_margin * self.scale)
        r_sq = r * r

        # Candidate points never leave the rings around home, so resources
        # at the enemy base cannot block anything.
        home = self.get_home(snap)
        if home is None or kind == "cc":
            reach_sq = float("inf")
            hx = hy = 0.0
        else:
            hx, hy = home
            reach_sq = (max(self.rings["barracks"][-1],
                            self.rings["depot"][-1]) + r) ** 2

        for res in snap.resources:
            if reach_sq == float("inf") or dist2(hx, hy, res.x, res.y) < reach_sq:
                out.append((res.x, res.y, r_sq))

        return out

    def free_point_near(self, snap, anchor, kind, resource_margin=None):
        """First legal spot on small rings around an anchor point."""
        if self.build_attempts.get(kind) is not None:
            return None

        obstacles = self.obstacles(snap, kind, resource_margin=resource_margin)
        failed = self.failed_points.setdefault(kind, {})
        loop = self.current_loop

        candidates = [anchor]
        for base_radius in (3.0, 5.0, 7.0):
            radius = base_radius * self.scale
            for i in range(8):
                angle = 2.0 * math.pi * i / 8.0
                candidates.append((
                    clamp(anchor[0] + radius * math.cos(angle),
                          self.edge, self.raw_resolution - self.edge),
                    clamp(anchor[1] + radius * math.sin(angle),
                          self.edge, self.raw_resolution - self.edge)))

        for point in candidates:
            when, count = failed.get(self.key(point), (-999999, 0))
            cooldown = min(FAILED_POINT_COOLDOWN * (2 ** max(0, count - 1)),
                           FAILED_POINT_COOLDOWN_MAX)
            if loop - when < cooldown:
                continue

            px, py = point
            if not any((px - ox) ** 2 + (py - oy) ** 2 < r_sq
                       for ox, oy, r_sq in obstacles):
                return point

        return None

    def find_build_point(self, snap, kind):
        # One outstanding attempt per kind.
        if self.build_attempts.get(kind) is not None:
            return None

        failed = self.failed_points[kind]
        obstacles = self.obstacles(snap, kind)
        loop = self.current_loop

        for point in self.candidate_points(snap, kind):
            if failed:
                when, count = failed.get(self.key(point), (-999999, 0))
                cooldown = min(FAILED_POINT_COOLDOWN * (2 ** max(0, count - 1)),
                               FAILED_POINT_COOLDOWN_MAX)
                if loop - when < cooldown:
                    continue

            px, py = point
            blocked = False

            for ox, oy, r_sq in obstacles:
                dx = px - ox
                dy = py - oy
                if dx * dx + dy * dy < r_sq:
                    blocked = True
                    break

            if not blocked:
                return point

        return None

    def start_build(self, kind, worker, point, loop, function_name):
        tag = worker.tag

        self.build_attempts[kind] = {
            "point": point,
            "loop": loop,
            "worker_tag": tag,
            "arrived": None,
        }

        self.busy_workers[tag] = loop + BUILDER_BUSY_LOOPS[kind]
        self.mark(kind, loop)
        self.orders[kind] += 1

        log("BUILD %s SCV=%d POS=(%.1f, %.1f)"
            % (kind.upper(), tag, point[0], point[1]))

        act = self.call(function_name, "now", [tag], list(point))

        if act is None:
            # Command was rejected - do not keep the slot or the worker.
            self.build_attempts[kind] = None
            self.busy_workers.pop(tag, None)
            self.orders[kind] -= 1

        return act

    BUILD_RESULT = {
        "depot": DEPOT_IDS,
        "barracks": (BARRACKS,),
        "bunker": (BUNKER,),
        "cc": BASE_IDS,
    }

    def check_build_attempts(self, snap, loop):
        for kind in ("depot", "barracks", "bunker", "cc"):
            attempt = self.build_attempts.get(kind)

            if attempt is None:
                continue

            px, py = attempt["point"]

            # The tolerance must stay below the minimum separation that
            # the obstacle check enforces, otherwise a NEIGHBOURING
            # building would be read as "our build succeeded" and a bad
            # point would never be blacklisted.
            existing = snap.of_type(*self.BUILD_RESULT[kind])
            tolerance_sq = self.build_tolerance_sq[kind]

            if any(dist2(px, py, u.x, u.y) < tolerance_sq for u in existing):
                self.build_attempts[kind] = None

                # Proven buildable ground - search near it next time.
                self.verified_ground.append((px, py))
                if len(self.verified_ground) > 6:
                    self.verified_ground.pop(0)
                continue

            worker = snap.scv_by_tag.get(attempt["worker_tag"])

            if worker is None:
                # Builder died on the way. The point is not to blame.
                self.build_attempts[kind] = None
                self.busy_workers.pop(attempt["worker_tag"], None)
                continue

            # Once the builder is standing on the spot and still nothing
            # has appeared, the placement is illegal - no need to wait out
            # the full walking timeout.
            if (attempt["arrived"] is None
                    and dist2(px, py, worker.x, worker.y) < self.arrived_radius_sq):
                attempt["arrived"] = loop

            deadline = (
                attempt["arrived"] + ARRIVED_TIMEOUT
                if attempt["arrived"] is not None
                else attempt["loop"] + BUILD_TIMEOUT
            )

            if loop > deadline:
                key = self.key(attempt["point"])
                count = self.failed_points[kind].get(key, (0, 0))[1]
                self.failed_points[kind][key] = (loop, count + 1)
                self.build_attempts[kind] = None
                self.busy_workers.pop(attempt["worker_tag"], None)

                log("BUILD FAILED: %s POS=(%.1f, %.1f) -> next point"
                    % (kind.upper(), px, py))

    # ------------------------------------------------------------------
    # ECONOMY
    # ------------------------------------------------------------------

    def supply_buffer(self, s):
        """Headroom to keep.

        A depot takes ~470 loops to finish, during which N barracks each
        eat a supply roughly every 400 loops, so the buffer has to scale
        with production capacity - otherwise the bot is permanently one
        depot behind and sits supply blocked half the game.
        """
        capacity = s["barracks"] + len(s["barracks_building"])
        return 4 + 3 * min(self.cfg["max_barracks"], capacity)

    def reserved_minerals(self, s):
        """Minerals held back from marine production.

        Six barracks can out-spend a single base, so without a reserve
        the marine queue eats the minerals that the depot - or the one
        Orbital Command upgrade that fixes the income gap in the first
        place - was waiting for. Marines are the only spender big enough
        to need holding back; everything else is far cheaper per step.
        """
        reserve = 0

        if (s["food_cap"] < 200
                and s["supply_left"] + s["supply_pending"] <= self.supply_buffer(s)):
            reserve += DEPOT_COST

        if (s["orbitals"] == 0 and s["barracks"] >= 1
                and s["workers"] >= min(15, self.cfg["worker_target"] - 2)
                and self.since("orbital", self.current_loop) > 700):
            reserve += ORBITAL_COST

        # FIX: marines ate every mineral, so the 400 for a second base was
        # never saved up. The main then mined out and the game was over -
        # with a full worker count and nothing to mine. The reserve times
        # out so that an expansion which can never be placed does not
        # freeze the whole economy.
        if self.expansion_wanted and self.since("expand_wanted",
                                                self.current_loop) < 4000:
            reserve += CC_COST

        return reserve

    def spendable(self, s):
        """Minerals free to spend on things that can wait."""
        return s["minerals"] - self.reserved_minerals(s)

    def issue_depot(self, snap, s, loop):
        if s["food_cap"] >= 200:
            return None

        if s["minerals"] < DEPOT_COST:
            return None

        # Depots under construction already cover this much supply.
        if s["supply_left"] + s["supply_pending"] > self.supply_buffer(s):
            return None

        if not self.ready("depot", loop):
            return None

        point = self.find_build_point(snap, "depot")
        if point is None:
            return None

        worker = self.pick_worker(s, point)
        if worker is None:
            return None

        return self.start_build("depot", worker, point, loop, "Build_SupplyDepot_pt")

    def issue_scv(self, snap, s, loop):
        if not s["cc"]:
            return None

        if s["workers"] >= self.cfg["worker_target"]:
            return None

        if s["minerals"] < SCV_COST:
            return None

        if s["supply_left"] < 1:
            return None

        if not self.ready("scv", loop):
            return None

        # Zero-queue policy: only a base that is not already training.
        for cc in s["cc"]:
            if cc.orders == 0:
                self.mark("scv", loop)
                self.orders["scv"] += 1
                return self.call("Train_SCV_quick", "now", [cc.tag])

        return None

    def mineral_patches(self, snap, home, loop):
        """Mineral fields at our own base, cached - they do not move.

        FIX: the cache used to compare its own length against itself,
        which can never invalidate. It now keys on how many resources the
        observation reports, so a patch that mines out drops the cache
        instead of leaving us ordering workers onto a dead unit tag.
        """
        resource_count = len(snap.resources)

        if (self._patch_cache is not None
                and resource_count == self._patch_cache_size
                and loop - self._patch_cache_loop < 4000):
            return self._patch_cache

        hx, hy = home
        patches = [
            u for u in snap.resources
            if u.unit_type in MINERAL_IDS
            and dist2(hx, hy, u.x, u.y) < self.patch_radius_sq
        ]

        # FIX: when the main mines out this list goes empty, and workers
        # then sit idle for the rest of the game with a full worker count
        # and zero income. Fall back to whatever minerals are left
        # anywhere we can see.
        if not patches:
            patches = [u for u in snap.resources if u.unit_type in MINERAL_IDS]

        patches.sort(key=lambda u: dist2(hx, hy, u.x, u.y))

        self._patch_cache = patches
        self._patch_cache_size = resource_count
        self._patch_cache_loop = loop

        return patches

    def return_idle_workers(self, snap, s, loop):
        """Send idle SCVs back to the minerals.

        FIX: an SCV that finishes a building goes idle next to it and, in
        the original, stayed idle for the rest of the game. With the queue
        check repaired this is now detectable, and it is a large part of
        why mineral income kept falling behind.
        """
        if not self.ready("harvest", loop):
            return None

        busy = self.busy_workers
        idle = [u for u in s["scvs"] if u.orders == 0 and busy.get(u.tag, -1) <= loop]

        if not idle:
            return None

        home = s["home"]
        if home is None:
            return None

        patches = self.mineral_patches(snap, home, loop)
        if not patches:
            return None

        # Rotate through the patches so returning workers do not all pile
        # onto the same one.
        patch = patches[self.orders["harvest"] % len(patches)]
        tags = [u.tag for u in idle]

        if self.scout_tag in tags:
            self.scout_tag = None

        self.mark("harvest", loop)
        self.orders["harvest"] += 1

        log("RETURN TO MINERALS: %d SCV" % len(tags))

        return self.call_first(
            ["Harvest_Gather_SCV_unit", "Harvest_Gather_unit"],
            "now", tags, patch.tag,
        )

    def directed_point(self, snap, distance):
        """A point `distance` tiles from home, towards the enemy."""
        home = self.get_home(snap)
        if home is None:
            return None

        target = self.enemy_point(snap)
        dx = target[0] - home[0]
        dy = target[1] - home[1]
        norm = max(1.0, math.hypot(dx, dy))

        edge = self.edge
        limit = self.raw_resolution - edge
        return (clamp(home[0] + dx / norm * distance, edge, limit),
                clamp(home[1] + dy / norm * distance, edge, limit))

    def issue_bunker(self, snap, s, loop):
        """Static defence on the approach to our base.

        Against the harder built-in bots the first attack arrives before
        a marine-only army can hold it in the open. A bunker turns four
        marines into a position that has to be broken instead of run
        over, which is the cheapest way to survive that timing.
        """
        wanted = self.cfg["bunkers"]

        if not wanted:
            return None

        existing = snap.of_type(BUNKER)
        if len(existing) >= wanted:
            return None

        if s["barracks"] < 1 or self.spendable(s) < BUNKER_COST:
            return None

        if not self.ready("bunker", loop):
            return None

        anchor = self.directed_point(snap, (9.0 + 3.0 * len(existing)) * self.scale)
        if anchor is None:
            return None

        point = self.free_point_near(snap, anchor, "bunker")
        if point is None:
            return None

        worker = self.pick_worker(s, point)
        if worker is None:
            return None

        return self.start_build("bunker", worker, point, loop,
                                "Build_Bunker_pt")

    def load_bunker(self, snap, s, loop):
        """Garrison the bunker when something is coming."""
        if not self.cfg["bunkers"] or not s["marines"]:
            return None

        if not s["threats_home"]:
            return None

        # Only garrison out of surplus. Four marines locked in a bunker
        # while the field army is down to six is how a defence collapses.
        if len(s["marines"]) <= self.cfg["home_reserve"]:
            return None

        if not self.ready("load", loop):
            return None

        for bunker in snap.of_type(BUNKER):
            if bunker.progress < 0.999 or bunker.cargo >= 4:
                continue

            free_slots = 4 - bunker.cargo
            tags = [m.tag for m in s["marines"][:free_slots]]

            if not tags:
                return None

            self.mark("load", loop)
            self.orders["load"] += 1

            log("LOAD BUNKER %d marines=%d" % (bunker.tag, len(tags)))

            return self.call_first(["Load_Bunker_unit", "Load_unit"],
                                   "now", tags, bunker.tag)

        return None

    def mineral_clusters(self, snap, home):
        """Group visible mineral fields into bases by proximity."""
        clusters = []

        for patch in snap.resources:
            if patch.unit_type not in MINERAL_IDS:
                continue

            for cluster in clusters:
                if (dist2(cluster[0][0], cluster[0][1], patch.x, patch.y)
                        < self.cluster_radius_sq):
                    cluster.append((patch.x, patch.y))
                    break
            else:
                clusters.append([(patch.x, patch.y)])

        out = []
        for cluster in clusters:
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            out.append(((cx, cy), len(cluster)))

        return out

    def home_patches(self, snap, s, loop):
        home = s["home"]
        if home is None:
            return []

        hx, hy = home
        return [u for u in snap.resources
                if u.unit_type in MINERAL_IDS
                and dist2(hx, hy, u.x, u.y) < self.patch_radius_sq]

    def wants_expansion(self, snap, s, loop):
        """Is a second base the right call right now?"""
        if self.cfg["expand_after_barracks"] is None:
            return False

        if len(s["cc"]) + len(s["cc_building"]) >= 2:
            return False

        # Running out of minerals at home makes this urgent rather than
        # optional: without it the game ends with 32 idle workers.
        if len(self.home_patches(snap, s, loop)) <= 3:
            return True

        return s["barracks"] >= self.cfg["expand_after_barracks"]

    def issue_expansion(self, snap, s, loop):
        """A second Command Center.

        One mineral line caps income at roughly sixteen workers, which is
        not enough to keep the harder profiles' barracks busy. This is
        the only mineral-only way to raise the ceiling.
        """
        if not self.wants_expansion(snap, s, loop):
            return None

        if s["minerals"] < CC_COST:
            return None

        # Never expand while the base is actually under attack. The extra
        # "wait until things have been calm for a while" rule only
        # applies when this is a greedy expansion; if the main is mining
        # out, waiting for calm means never expanding at all.
        if s["threats_home"]:
            return None

        urgent = len(self.home_patches(snap, s, loop)) <= 3

        if not urgent and self.since("defend", loop) < 600:
            return None

        if not self.ready("expand", loop):
            return None

        home = s["home"]
        if home is None:
            return None

        # Nearest mineral cluster that is not the one we are already on.
        candidates = [
            (centre, size) for centre, size in self.mineral_clusters(snap, home)
            if dist2(home[0], home[1], centre[0], centre[1]) > self.expansion_min_sq
            and size >= 4
        ]

        if not candidates:
            return None

        centre = min(candidates,
                     key=lambda c: dist2(home[0], home[1], c[0][0], c[0][1]))[0]

        # A Command Center has to sit CLOSE to the minerals - the patch
        # ring is only ~6.5 tiles across, so the usual "keep away from
        # resources" margin makes every candidate illegal and no
        # expansion is ever placed.
        point = self.free_point_near(snap, centre, "cc", resource_margin=1.6)
        if point is None:
            return None

        worker = self.pick_worker(s, point)
        if worker is None:
            return None

        log("EXPAND to (%.1f, %.1f)" % (point[0], point[1]))

        return self.start_build("cc", worker, point, loop,
                                "Build_CommandCenter_pt")

    def issue_orbital(self, snap, s, loop):
        """Morph the Command Center into an Orbital Command.

        A single base with sixteen workers cannot pay for six barracks -
        that mineral gap is what kept the bot supply blocked. MULEs close
        it: 50 energy mines roughly 225 extra minerals, which is about a
        25% income increase for a one-off 150 mineral investment.
        """
        if s["barracks"] < 1:
            return None

        if s["workers"] < 15:
            return None

        if s["minerals"] < ORBITAL_COST:
            return None

        if not self.ready("orbital", loop):
            return None

        for base in s["cc"]:
            if base.unit_type == CC and base.orders == 0:
                self.mark("orbital", loop)
                self.orders["orbital"] += 1
                log("MORPH ORBITAL COMMAND CC=%d" % base.tag)
                return self.call("Morph_OrbitalCommand_quick", "now", [base.tag])

        return None

    def call_mule(self, snap, s, loop):
        """Drop a MULE on the closest mineral patch."""
        if not self.ready("mule", loop):
            return None

        orbital = None
        for base in s["cc"]:
            if base.unit_type == ORBITAL and base.energy >= MULE_ENERGY:
                orbital = base
                break

        if orbital is None or s["home"] is None:
            return None

        patches = self.mineral_patches(snap, s["home"], loop)
        if not patches:
            return None

        self.mark("mule", loop)
        self.orders["mule"] += 1

        log("CALLDOWN MULE energy=%d" % orbital.energy)

        patch = patches[0]

        act = self.call("Effect_CalldownMULE_unit", "now",
                        [orbital.tag], patch.tag)

        if act is None:
            # The point variant takes coordinates, not a unit tag.
            act = self.call("Effect_CalldownMULE_pt", "now",
                            [orbital.tag], [patch.x, patch.y])

        return act

    def lower_depots(self, snap, s, loop):
        """Lower finished depots.

        A raised depot is a wall. With depots ringing the base, leaving
        them up fences our own army and workers in - so every completed
        one is lowered as soon as it finishes.
        """
        if not self.ready("lower", loop):
            return None

        raised = [d.tag for d in snap.of_type(DEPOT) if d.progress >= 0.999]

        if not raised:
            return None

        self.mark("lower", loop)
        self.orders["lower"] += 1

        return self.call("Morph_SupplyDepot_Lower_quick", "now", raised)

    def set_rally(self, snap, s, loop):
        """Point each finished barracks at the staging position once."""
        new = [
            b.tag for b in s["barracks_all"]
            if b.unit_type == BARRACKS and b.progress >= 0.999
            and b.tag not in self.rallied
        ]

        if not new:
            return None

        act = self.call_first(
            ["Rally_Units_pt", "Rally_Building_pt"],
            "now", new, list(self.guard_point(snap)))

        if act is not None:
            self.rallied.update(new)
            log("SET RALLY barracks=%s" % new)

        return act

    # ------------------------------------------------------------------
    # PRODUCTION
    # ------------------------------------------------------------------

    def issue_barracks(self, snap, s, loop):
        if s["barracks"] + len(s["barracks_building"]) >= self.cfg["max_barracks"]:
            return None

        # Respects the reserve: an eighth barracks is worth less than the
        # second base or the depot that stops a supply block.
        if self.spendable(s) < BARRACKS_COST:
            return None

        if not self.ready("barracks", loop):
            return None

        point = self.find_build_point(snap, "barracks")
        if point is None:
            return None

        worker = self.pick_worker(s, point)
        if worker is None:
            return None

        return self.start_build("barracks", worker, point, loop, "Build_Barracks_pt")

    def issue_marine(self, snap, s, loop):
        if not s["barracks_all"]:
            return None

        if s["minerals"] < MARINE_COST:
            return None

        if s["supply_left"] < 1:
            return None

        if not self.ready("marine", loop):
            return None

        # Only train as many as we can actually pay for and house, so we
        # do not burn a step on commands SC2 will reject.
        spendable = s["minerals"] - self.reserved_minerals(s)
        affordable = min(spendable // MARINE_COST, s["supply_left"])
        if affordable < 1:
            return None

        # Zero-queue policy - now actually enforced, see queue().
        tags = []
        for b in s["barracks_all"]:
            if b.unit_type == BARRACKS and b.progress >= 0.999 and b.orders == 0:
                tags.append(b.tag)
                if len(tags) >= affordable:
                    break

        if not tags:
            return None

        self.mark("marine", loop)
        self.orders["marine"] += len(tags)

        return self.call("Train_Marine_quick", "now", tags)

    # ------------------------------------------------------------------
    # COMBAT
    # ------------------------------------------------------------------

    TARGET_PRIORITY = {
        105: 0,   # Zergling
        9: 0,     # Baneling
        110: 1,   # Roach
        107: 1,   # Hydralisk
        108: 1,   # Mutalisk
        126: 2,   # Queen
        98: 3,    # Spine Crawler
        99: 3,    # Spore Crawler
        104: 4,   # Drone
        89: 5,    # Spawning Pool
        86: 6,    # Hatchery
        100: 6,   # Lair
        101: 6,   # Hive
    }

    def prune_dead_targets(self, loop):
        for tag in [t for t, until in self.dead_targets.items() if until <= loop]:
            del self.dead_targets[tag]

    def pick_target(self, s, origin):
        """Highest-priority enemy, tie-broken by distance to our army.

        FIX: the original ranked by global lowest health, which happily
        sent the whole army across the map after a wounded straggler, and
        its priority table listed drones/hatcheries that had already been
        filtered out of the candidate list.
        """
        enemies = s["enemy_all"]

        if not enemies:
            return None

        # FIX: the army spent five minutes re-issuing an attack on the
        # same tag without anything happening. A remembered snapshot of a
        # structure that is already dead reads exactly like a live target
        # in the raw observation, so prefer what we can actually see, and
        # give up on a target that will not die.
        alive = [u for u in enemies if u.tag not in self.dead_targets]

        if alive:
            enemies = alive

        visible = [u for u in enemies if u.visible]

        if visible:
            enemies = visible

        priority = self.TARGET_PRIORITY

        if origin is None:
            return min(enemies, key=lambda u: priority.get(u.unit_type, 7))

        ox, oy = origin
        return min(
            enemies,
            key=lambda u: (priority.get(u.unit_type, 7), dist2(ox, oy, u.x, u.y)),
        )

    def attack_with(self, tags, target, label):
        log("SC2 ACTION: %s MARINES=%d TARGET=%d" % (label, len(tags), target.tag))

        act = self.call("Attack_unit", "now", tags, target.tag)

        if act is not None:
            return act

        # Target-unit command rejected - attack its position instead.
        return self.call("Attack_pt", "now", tags, [target.x, target.y])

    def defend_home(self, snap, s, loop):
        """Checked every step, regardless of army size or attack cooldown.

        Without this the whole army can be off attacking while zerglings
        eat the depots and SCVs at home.

        FIX: this used to pull EVERY living marine regardless of where
        they were on the map - so a single threat near base recalled the
        entire army even mid-attack on the enemy base. On harder
        profiles this fires far more often (more frequent harassment),
        so pushes almost never survived long enough to finish. Marines
        already far away and committed to an attack now stay committed;
        only marines already near home respond.
        """
        marines = s["marines"]
        threats = s["threats_home"]

        if not threats:
            return None

        # RECOVERY FIX: OBRANA must stay actionable even when the first
        # defensive fight has reduced the marine count to zero. If a
        # completed barracks exists, immediately replace one marine before
        # falling through to the normal regroup/attack logic. This prevents
        # the MCDM layer from getting stuck in PROIZVODNJA while the base is
        # being overrun.
        if not marines:
            return self.issue_marine(snap, s, loop)

        if not self.ready("defend", loop):
            return None

        hx, hy = s["home"]

        local_radius_sq = self.defense_radius_sq * 4.0
        nearby = [
            m for m in marines
            if dist2(hx, hy, m.x, m.y) < local_radius_sq
        ]

        if not nearby:
            return None

        threat = min(threats, key=lambda u: dist2(hx, hy, u.x, u.y))

        # FIX: charging out with fewer marines than there are attackers
        # feeds them in one at a time and loses every trade. Below that
        # bar, gather at the guard point instead and keep producing -
        # massed marines next to the base (and its SCVs) at least trade.
        needed = min(len(threats), self.cfg["home_reserve"])

        if len(nearby) < max(2, needed):
            if not self.ready("move", loop):
                return None

            point = self.guard_point(snap)

            self.mark("move", loop)
            self.orders["regroup"] += 1

            log("REGROUP: %d marinaca vs %d prijetnji"
                % (len(nearby), len(threats)))

            return self.call("Move_pt", "now",
                             [m.tag for m in nearby], list(point))

        self.mark("defend", loop)
        self.orders["defend"] += 1

        return self.attack_with([m.tag for m in nearby], threat, "DEFEND_HOME")

    def attack(self, snap, s, loop):
        marines = s["marines"]

        cfg = self.cfg
        minimum = (
            cfg["attack_min_napad"] if self.action == "NAPAD"
            else cfg["attack_min"]
        )

        if len(marines) < minimum:
            return None

        if not self.ready("attack", loop):
            return None

        # Hold a reserve at home instead of committing every marine.
        attackers = marines[cfg["home_reserve"]:]

        # FIX: the threshold describes the STRIKE FORCE, not the total
        # army. Attacking a defended base with attack_min minus the home
        # reserve is how a won game turns into a lost one: the push dies,
        # and the counter-attack arrives while we are rebuilding.
        if len(attackers) < max(cfg["min_attackers"], minimum):
            return None

        tags = [m.tag for m in attackers]

        sum_x = 0.0
        sum_y = 0.0
        for m in attackers:
            sum_x += m.x
            sum_y += m.y
        origin = (sum_x / len(attackers), sum_y / len(attackers))

        target = self.pick_target(s, origin)

        # Same target for too long means we are not actually hurting it.
        if target is not None:
            if self.attack_target != target.tag:
                self.attack_target = target.tag
                self.attack_target_since = loop
            elif loop - self.attack_target_since > STUCK_TARGET_LOOPS:
                log("[TARGET] %d se ne mice %d loopova - preskacem ga"
                    % (target.tag, loop - self.attack_target_since))
                self.dead_targets[target.tag] = loop + DEAD_TARGET_COOLDOWN
                self.attack_target = None
                target = self.pick_target(s, origin)

        self.mark("attack", loop)
        self.orders["attack"] += 1

        if target is not None:
            return self.attack_with(tags, target, "ATTACK_UNIT")

        point = self.enemy_point(snap)

        log("SC2 ACTION: ATTACK_POINT MARINES=%d POS=(%.1f, %.1f)"
            % (len(tags), point[0], point[1]))

        return self.call("Attack_pt", "now", tags, list(point))

    def retreat(self, snap, s, loop):
        """Pull a losing field army back to the base."""
        marines = s["marines"]
        home = s["home"]

        if not marines or home is None or s["threats_home"]:
            return None

        if not self.ready("move", loop):
            return None

        hx, hy = home
        far = [m for m in marines
               if dist2(hx, hy, m.x, m.y) > self.defense_radius_sq]

        if len(far) < 3:
            return None

        point = self.rally_point(snap)

        self.mark("move", loop)
        self.orders["retreat"] += 1

        log("SC2 ACTION: RETREAT MARINES=%d" % len(far))

        return self.call("Move_pt", "now", [m.tag for m in far], list(point))

    def rally(self, snap, s, loop):
        marines = s["marines"]

        if not marines or len(marines) >= self.cfg["attack_min_napad"]:
            return None

        # Never Move marines that are fighting - a fresh Move order
        # cancels the in-progress Attack and they disengage mid-fight.
        if self.since("attack", loop) < 96 or self.since("defend", loop) < 96:
            return None

        if not self.ready("move", loop):
            return None

        point = self.rally_point(snap)

        self.mark("move", loop)
        self.orders["move"] += 1

        return self.call("Move_pt", "now", [m.tag for m in marines], list(point))

    def scout(self, snap, s, loop):
        """Send a single SCV to the mirrored start location."""
        if self.scout_tag is not None:
            if self.scout_tag in snap.scv_by_tag:
                return None
            self.scout_tag = None

        if s["workers"] < self.cfg["scout_min_workers"]:
            return None

        if not self.ready("scout", loop):
            return None

        point = self.enemy_point(snap)
        worker = self.pick_worker(s, point)

        if worker is None:
            return None

        self.scout_tag = worker.tag
        self.mark("scout", loop)
        self.orders["scout"] += 1

        log("SC2 ACTION: SCOUT SCV=%d POS=(%.1f, %.1f)"
            % (self.scout_tag, point[0], point[1]))

        return self.call("Move_pt", "now", [self.scout_tag], list(point))

    # ------------------------------------------------------------------
    # DECISION LOGGING
    # ------------------------------------------------------------------

    def log_decision(self, s, decision):
        if not VERBOSE:
            return

        result = decision.result
        rank_of = {index: rank for rank, index in enumerate(result.ranking, 1)}

        print()
        print("=" * 78)
        print("MCDM  loop %d  |  method=%s  normalisation=%s%s"
              % (self.current_loop, decision.method, decision.scheme_used,
                 "  v=%.2f" % decision.v_used if decision.method == "vikor" else ""))
        print("=" * 78)

        header = "%-13s" % "alternativa"
        for name in CRITERIA:
            header += "%9s" % name[:8]
        header += "%9s%6s" % ("score", "rank")
        print(header)

        for i, name in enumerate(ACTIONS):
            row = "%-13s" % name[:12]
            for value in decision.matrix[i]:
                row += "%9.2f" % value

            if i in result.scores:
                row += "%9.3f%6d" % (result.scores[i], rank_of[i])
            else:
                row += "%9s%6s" % ("-", "n/a")

            print(row)

        if decision.method == "vikor":
            meta = result.meta
            print("VIKOR: C1=%s C2=%s kompromisni skup=%s"
                  % (meta["c1"], meta["c2"],
                     [ACTIONS[i] for i in meta["compromise"]]))

        print("ODLUKA: %s" % decision.action)
        print(
            "STANJE: SCV=%d MARINCI=%d BARAKE=%d MIN=%d SUPPLY=%d/%d "
            "NEPRIJATELJ=%d PRIJETNJA=%d"
            % (s["workers"], s["army"], s["barracks"], s["minerals"],
               s["food_used"], s["food_cap"], s["enemy"],
               len(s["threats_home"])))

    # ------------------------------------------------------------------
    # MAIN STEP
    # ------------------------------------------------------------------

    def step(self, timestep):
        if getattr(timestep, "first", lambda: False)():
            episode = self.episode + 1
            self.reset()
            self.episode = episode

        obs = timestep.observation

        raw_loop = getattr(obs, "game_loop", 0)
        try:
            loop = int(raw_loop[0])
        except (TypeError, IndexError, KeyError):
            try:
                loop = int(raw_loop)
            except (TypeError, ValueError):
                loop = 0

        self.current_loop = loop
        self.prune_busy(loop)
        self.prune_dead_targets(loop)

        snap = Snapshot(getattr(obs, "raw_units", ()))

        if not self.calibrated:
            self.calibrate(snap)

        s = self.state(snap, obs.player)

        wanted = self.wants_expansion(snap, s, loop)

        if wanted and not self.expansion_wanted:
            self.mark("expand_wanted", loop)

        self.expansion_wanted = wanted

        self.check_build_attempts(snap, loop)

        # The scout has arrived (or died) - release it back to mining.
        if self.scout_tag is not None and self.since("scout", loop) > 400:
            self.scout_tag = None

        # ---------------- MCDM decision ----------------
        if self.mcdm is not None and self.ready("decision", loop):
            self.mark("decision", loop)

            decision = self.mcdm.decide(s)
            self.action = decision.action
            self.decisions += 1
            # Cumulative across episodes, unlike self.decisions which is
            # per-episode state - mixing the two produced C2 = 200%.
            self.orders["decisions"] += 1

            if decision.method == "vikor":
                meta = decision.result.meta
                self.orders["vikor_c1"] += int(bool(meta.get("c1")))
                self.orders["vikor_c2"] += int(bool(meta.get("c2")))
                self.orders["vikor_set"] += len(meta.get("compromise", [1]))

            if self.decision_log is not None:
                self.decision_log.write(self.episode, loop, decision, s)

            self.log_decision(s, decision)

        # ---------------- hard constraints ----------------
        # These outrank MCDM: being supply blocked or letting the base
        # burn down cannot be traded away by a strategic preference.
        act = self.issue_depot(snap, s, loop)

        if act is None:
            act = self.defend_home(snap, s, loop)

        if act is None:
            act = self.load_bunker(snap, s, loop)

        # ---------------- MCDM dispatch ----------------
        if act is None:
            for handler in self.DISPATCH.get(self.action, ()):
                act = handler(self, snap, s, loop)
                if act is not None:
                    break

        # ---------------- fallback chain ----------------
        # A step is never wasted, and the attack can never be starved by
        # the strategic layer.
        if act is None:
            for handler in self.FALLBACK:
                act = handler(self, snap, s, loop)
                if act is not None:
                    break

        self.next_step_mul = self.plan_step_mul(s, act is not None)

        return act if act is not None else self.no_op()

    def plan_step_mul(self, s, acted):
        """Game loops the runner should advance after this step.

        Idle steps are the majority of an episode, and each one costs a
        full observation round trip. When nothing needed doing and no
        enemy army is on screen, several loops can safely be collapsed
        into one step; anything that needs a reaction drops straight back
        to the base rate.
        """
        base = self.base_step_mul

        if not self.idle_speedup:
            return base

        if acted or s["enemy_army"] or s["threats_home"]:
            return base

        if self.build_attempts["depot"] is not None:
            return base
        if self.build_attempts["barracks"] is not None:
            return base

        return min(base * IDLE_STEP_FACTOR, MAX_STEP_MUL)

    DISPATCH = {
        "EKONOMIJA": (call_mule, issue_scv, issue_expansion, issue_orbital,
                      return_idle_workers),
        "PROIZVODNJA": (issue_barracks, issue_marine, set_rally),
        "NAPAD": (attack, issue_marine),
        "OBRANA": (load_bunker, defend_home, issue_bunker, issue_marine),
        "IZVIDANJE": (scout,),
        "POVLACENJE": (retreat,),
        "ODRZAVANJE": (call_mule, lower_depots, return_idle_workers,
                       issue_marine, rally),
    }

    FALLBACK = (
        issue_expansion,
        issue_scv,
        issue_bunker,
        issue_orbital,
        issue_barracks,
        attack,
        issue_marine,
        call_mule,
        set_rally,
        lower_depots,
        return_idle_workers,
        rally,
    )


# ======================================================================
# MAIN
# ======================================================================

def print_summary(agent, results):
    print()
    print("=" * 60)
    print("SIMULATION FINISHED")
    print("=" * 60)

    labels = [
        ("Depots", "depot"),
        ("Barracks", "barracks"),
        ("SCVs", "scv"),
        ("Marines", "marine"),
        ("Attacks", "attack"),
        ("Defends", "defend"),
        ("Retreats", "retreat"),
        ("Moves", "move"),
        ("Scouts", "scout"),
        ("Harvest", "harvest"),
        ("Orbitals", "orbital"),
        ("MULEs", "mule"),
        ("Regroups", "regroup"),
        ("Bunkers", "bunker"),
        ("Expansions", "cc"),
    ]

    for label, key in labels:
        print("%-11s: %d" % (label, agent.orders[key]))

    if agent.orders["vikor_set"]:
        decisions = max(1, agent.orders["decisions"])
        print("%-11s: C1 %.0f%%, C2 %.0f%%, prosj. kompromisni skup %.2f"
              % ("VIKOR",
                 100.0 * agent.orders["vikor_c1"] / decisions,
                 100.0 * agent.orders["vikor_c2"] / decisions,
                 agent.orders["vikor_set"] / float(decisions)))

    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    ties = sum(1 for r in results if r == 0)

    print("%-11s: %d W / %d L / %d T" % ("Result", wins, losses, ties))

    print()
    print(
        "THESIS_METRICS,"
        "method=%s,profile=%s," % (agent.method_name, agent.profile_name) +
        "depots=%d,barracks=%d,scvs=%d,marines=%d,attacks=%d,moves=%d,"
        "defends=%d,retreats=%d,scouts=%d,orbitals=%d,mules=%d,"
        "episodes=%d,wins=%d,losses=%d"
        % (
            agent.orders["depot"], agent.orders["barracks"], agent.orders["scv"],
            agent.orders["marine"], agent.orders["attack"], agent.orders["move"],
            agent.orders["defend"], agent.orders["retreat"], agent.orders["scout"],
            agent.orders["orbital"], agent.orders["mule"],
            len(results), wins, losses,
        )
    )


def main(_):
    global VERBOSE
    VERBOSE = not FLAGS.quiet

    profile = FLAGS.profile or PROFILE_FOR_DIFFICULTY.get(
        FLAGS.difficulty, "easy")

    if profile not in PROFILES:
        raise ValueError("unknown profile %r, expected one of %s"
                         % (profile, sorted(PROFILES)))

    map_name = resolve_map(FLAGS.map, FLAGS.map_dir)

    print("=" * 60)
    print("MCDM TERRAN MARINE PUSH")
    print("Terran vs Zerg (%s) | %s | step_mul=%d"
          % (FLAGS.difficulty, map_name, FLAGS.step_mul))
    print("Profil: %s | %s" % (profile, PROFILES[profile]))
    print("Metoda: %s | normalizacija: %s%s"
          % (FLAGS.mcdm, FLAGS.normalization,
             " | v=%.2f" % FLAGS.vikor_v if FLAGS.mcdm == "vikor" else ""))
    print("=" * 60)

    interface = features.AgentInterfaceFormat(
        use_raw_units=True,
        raw_resolution=FLAGS.raw_resolution,
        action_space=actions.ActionSpace.RAW,
    )

    players = [
        sc2_env.Agent(sc2_env.Race.terran, name="MCDM_Terran"),
        sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty[FLAGS.difficulty]),
    ]

    scheme = None if FLAGS.normalization == "classic" else "linear"

    decision_log = DecisionLog(FLAGS.decision_log) if FLAGS.decision_log else None

    agent = WinAgent(step_mul=FLAGS.step_mul,
                     idle_speedup=FLAGS.idle_speedup,
                     method=FLAGS.mcdm,
                     scheme=scheme,
                     vikor_v=FLAGS.vikor_v,
                     decision_log=decision_log,
                     profile=profile,
                     raw_resolution=FLAGS.raw_resolution)
    results = []
    env_steps = 0

    with sc2_env.SC2Env(
        map_name=map_name,
        players=players,
        random_seed=(None if FLAGS.random_seed < 0 else FLAGS.random_seed),
        agent_interface_format=interface,
        step_mul=FLAGS.step_mul,
        realtime=False,
        visualize=FLAGS.visualize,
        game_steps_per_episode=25000,
        score_index=-1,
        disable_fog=False,
        ensure_available_actions=False,
    ) as env:
        try:
            for episode in range(FLAGS.episodes):
                timesteps = env.reset()

                while True:
                    action = agent.step(timesteps[0])

                    # Ask the environment to advance a variable number of
                    # loops. Older pysc2 builds do not accept the keyword,
                    # in which case the fixed rate is used.
                    try:
                        timesteps = env.step([action],
                                             step_mul=agent.next_step_mul)
                    except TypeError:
                        timesteps = env.step([action])

                    env_steps += 1

                    if timesteps[0].last():
                        break

                reward = float(timesteps[0].reward)
                results.append(reward)

                print("EPISODE %d FINISHED, reward=%.0f, env steps=%d"
                      % (episode + 1, reward, env_steps))

        finally:
            print_summary(agent, results)

            if decision_log is not None:
                decision_log.close()
                print("Decision log: %s" % FLAGS.decision_log)


if __name__ == "__main__":
    app.run(main)