# Exploratory Data Analysis Report (BotPrint)

## Scope
This report summarises exploratory analysis of `dataset/processed/cleaned_events.csv`.
EDA is descriptive: **visual or summary differences do not prove that a feature can distinguish humans and bots**.

## Sample sizes
- Human: 665 events across 9 sessions.
- Bot: 531 events across 15 sessions.
- Bot subtypes (session-level):
- CoordinatedBot: 325 events across 5 sessions
- MimicBot: 67 events across 5 sessions
- RandomBot: 139 events across 5 sessions

Session-level conclusions in this EDA are based on session-level aggregates, not by treating individual events as independent sessions.

## Observed patterns (descriptive)
- **Event frequency:** session-level events-per-session and events-per-second summaries show how active sessions are; distributions overlap and should be interpreted with session counts in mind.
- **Mouse movement:** event-level `Speed`/`Acceleration`/displacement distributions and session-level speed variability (CV/IQR) were examined where mouse events exist.
- **Keyboard typing:** inter-key timing/count fields were summarised when present; coverage depends on keyboard event availability.
- **Click behaviour:** click intervals/counts were summarised when present; session-level medians reduce the influence of unequal event counts.
- **Scroll behaviour:** scroll delta/speed/count were examined where scroll events exist.
- **Session duration:** compared at session level; bot subtypes show descriptive differences but small subtype samples.
- **Idle time:** session-level mean/median/standard deviation of `IdleMs` characterise pauses.
- **Navigation patterns:** navigation coverage may be limited; sparse navigation should not be over-interpreted.
- **Behavioural variability:** interpretable within-session variability measures (std/IQR/CV) were used to describe regularity/irregularity.

## Potential behavioural differences to investigate further
These are hypotheses, not confirmed discriminators:
- Timing regularity (mouse speed variability, inter-key interval variability, click interval variability).
- Activity pacing (events per second, idle-time distribution) at session level.
- Movement dynamics (speed/acceleration distributions) aggregated per session.
- Bot subtype differences (CoordinatedBot vs RandomBot vs MimicBot) using larger samples and controlled comparisons.

## Limitations
- **Small session counts**, especially per bot subtype (5 sessions each), limit statistical stability and generalisability.
- **Event dependence:** events are nested within sessions; event-level plots can over-represent sessions with many events.
- **Coverage differences:** not all sessions contain all event types (mouse/keyboard/click/scroll/navigation).
- **Missing values are structural** for some event types in the raw/cleaned event schema; summaries use available non-missing values and report counts where relevant.
- **Collection/context effects:** observed differences may partly reflect task design, instrumentation, or scripted scenarios rather than intrinsic human/bot behaviour.
- **No causal claims:** EDA identifies patterns worth testing, not causal mechanisms.

## Features worth investigating further (session-level)
- Mouse: median/IQR/CV of speed and acceleration; path-length proxies from DX/DY; movement burstiness.
- Keyboard: median/IQR/CV of inter-key intervals; typing-event rate; pause patterns.
- Click: median/IQR of click intervals; click rate; double-click/burst indicators if derivable.
- Scroll: median scroll delta/speed; scroll frequency and direction changes.
- Temporal: session duration, idle-time mean/median/std, active-time ratio.
- Navigation: navigation count/rate where available.
- Cross-modal variability: compare regularity across modalities within the same session.

## Reproducibility and data integrity
- Analysis uses `dataset/processed/cleaned_events.csv` produced by the preprocessing pipeline.
- Original raw data in `dataset/raw/` was not modified.
- Labels/identifiers are not used as behavioural features in downstream modelling; EDA uses `UserType`/`BotType` only for descriptive grouping.
