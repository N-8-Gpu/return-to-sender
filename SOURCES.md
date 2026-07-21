# SOURCES.md: citation registry for Return to Sender
Every constant in `config.py` cites an entry here by tag, or is marked ASSUMPTION with a range.
Rule: a number with no source is a hypothesis, and hypotheses get sensitivity sweeps, not confidence.

## Verified sources

**[S1] Calgary landfill battery fires (50+).** Calgary Fire Department, reported in LiveWire Calgary, "Calgary Fire Department urges caution with lithium-ion batteries ahead of holiday season," Nov 6, 2024. https://livewirecalgary.com/2024/11/06/calgary-fire-department-urges-caution-with-lithium-ion-batteries-ahead-of-holiday-season/
Feeds: fire-channel calibration target (~10-12 fires/quarter regional scale).

**[S2] ~40 fires/year; tour observations (compactor, cover, Freon dating, ecocentre streams).** City of Calgary landfill education staff, site tour, July 2026 (personal communication).
Feeds: fire calibration cross-check; narrative anchors.

**[S3] Battery-crushing fire mechanism (official).** City of Calgary, "Safe battery disposal," calgary.ca/waste/residential/safe-battery-disposal.html; CFD "Damaged Batteries Should Stay Dead" campaign, newsroom.calgary.ca, Nov 2024.
Feeds: mechanism claim on poster and invoice preamble.

**[S4] Continental fire damage: ~$2.5B (2025), 448 incidents, ~100 catastrophic at $500k to tens of millions; Vape Effect +26%.** R. Fogelman / Fire Rover, 9th Annual Waste & Recycling Facility Fires Report US/CAN (2026); summarized in Resource Recycling, Mar 27, 2026. https://resource-recycling.com/e-scrap/2026/03/27/report-pegs-fire-losses-at-2-5b-in-us-and-canada-recycling-industry/
Feeds: u_F (cost per fire) range for sensitivity sweep; impact section.

**[S5] 1-in-4 waste-facility battery fires cause service disruption and six-figure damage (federal survey).** InvestigateTV, May 11, 2026. https://www.investigatetv.com/2026/05/11/hidden-heat-battery-disposal-gaps-keep-fueling-fires-states-consider-who-should-foot-bill/
Feeds: escalation-probability assumption bounds.

**[S6] "Producers get off scot-free" (industry fire-protection lead).** R. Fogelman, Fire Rover, 2021. https://firerover.com/li-ion-battery-fires-unfairly-cost-waste-recycling-and-scrap-operators-over-1-2-billion-annually/
Feeds: pitch quote; problem framing. Includes Eunomia UK estimate (~£158M/yr) as a comparable.

**[S7] 5,000+ recycling-facility fires per year (industry estimate).** National Waste & Recycling Association (2024), via Resource Recycling, May 13, 2025. https://resource-recycling.com/recycling/2025/05/13/a-hot-topic-for-recyclers-battery-related-fires/
Feeds: sensor-layer addressable-event pool.

**[S8] Landfills = 17% of Canada's methane, 2.8% of GHGs (2022).** Environment and Climate Change Canada, QP Note ECCC-QP-000012 (Sep 2025). https://search.open.canada.ca/qpnotes/record/ec%2CECCC-QP-000012 (2021 figure 19%: Canada Gazette Pt.1 Vol.158 No.26.)
Feeds: context stat; not a model constant.

**[S9] Federal Landfill Methane Regulations (~50% cut by 2030).** ECCC, canada.ca "Landfill Methane Regulations."
Feeds: policy-tailwind claim.

**[S10] EU battery passport mandatory 18 Feb 2027; EV/LMT/industrial >2 kWh only; small consumer batteries excluded from that wave.** Regulation (EU) 2023/1542, Art. 77; plain-language: Battery Pass Consortium, thebatterypass.eu.
Feeds: gap argument; future-work section.

**[S11] Call2Recycle: 6.8M kg collected 2024 (+17%); 50M+ kg cumulative; programs in six provinces incl. NS; PRO in ON and AB; 89% of Canadians within 15 km of drop-off; Recycle Your Vapes program exists.** Call2Recycle Canada 2024 Annual Report release, Jun 24, 2025. https://call2recycle.ca
Feeds: capture-rate prior; "collection exists, accounting doesn't" positioning; targeted-deployment partner.

**[S12] Battery waste audits already run: Tetra Tech waste-composition studies, ~0.2 kg batteries disposed per capita (BC).** Call2Recycle BC Annual Report 2024, filed with BC government. https://www2.gov.bc.ca/assets/gov/environment/waste-management/recycling/recycle/batteries-call2recycle/c2r_batteries_annual_and_assurance_reports_2024.pdf
Feeds: audit-channel realism (n, detection d); per-capita disposal anchor.

**[S13] NS deposit return rate ~80% (among Canada's best); depot within 20 km of 90% of Nova Scotians.** Divert NS, divertns.ca.
Feeds: achievable-capture ceiling in intervention scenarios (beta bound).

**[S14] NS EPR launched Dec 1, 2025 (costs shift municipalities to producers); Divert NS 2026 producer fee consultation underway.** Divert NS, divertns.ca.
Feeds: customer/implementation path; no model constant.

**[S15] Fire Rover product (existing sensor comparator).** firerover.com.
Feeds: section D honesty; sensor differentiation.

## Unverified: use only with ASSUMPTION tag + sensitivity sweep
- **[U1] Calgary-specific cost per landfill fire.** No public figure found; use S4 continental range, labeled continental. The absence is itself an argument: nobody measures, so nobody bills.
- **[U2] FCM "municipalities collect ~10 cents of each tax dollar."** Attribute to FCM pending confirmation at fcm.ca.
- **[U3] Provincial legal authority for damage-based modulated category fees.** Unverified; listed as open question.
- **[U4] Per-tonne landfill perpetual-care cost (u_LF).** ASSUMPTION range pending a municipal financial report (PSAB landfill liability note is the place to look).
