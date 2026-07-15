suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(scales)
  library(ggrepel)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) normalizePath(args[[1]], winslash = "/", mustWork = TRUE) else normalizePath(".", winslash = "/", mustWork = TRUE)
data_root <- file.path(root, "data")
main_dir <- file.path(root, "figures", "main")
supp_dir <- file.path(root, "figures", "supplementary")
source_dir <- file.path(data_root, "figure_source_data")
dir.create(main_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(supp_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)

pal <- c(
  ink = "#252A30", neutral = "#777C82", light = "#DDE1E5",
  blue = "#3A78A8", teal = "#2A9D8F", orange = "#D8893C",
  red = "#C7524A", purple = "#8064A2", yellow = "#E2B64B"
)
algorithm_colors <- c(
  nsga2 = pal[["blue"]], spea2 = pal[["orange"]],
  rvea = pal[["teal"]], moead = pal[["purple"]]
)

theme_pub <- function(base_size = 7) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = pal[["ink"]]),
      axis.ticks = element_line(linewidth = 0.35, colour = pal[["ink"]]),
      axis.title = element_text(size = base_size, colour = pal[["ink"]]),
      axis.text = element_text(size = base_size - 0.5, colour = pal[["ink"]]),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 0.5),
      strip.background = element_rect(fill = "#F3F5F6", colour = NA),
      strip.text = element_text(size = base_size, face = "bold", margin = margin(2, 2, 2, 2)),
      plot.title = element_text(size = base_size + 0.6, face = "bold", colour = pal[["ink"]]),
      plot.subtitle = element_text(size = base_size - 0.3, colour = pal[["neutral"]]),
      plot.tag = element_text(size = 9, face = "bold"),
      panel.grid.major.y = element_line(linewidth = 0.22, colour = "#E8EAEC"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(4, 5, 4, 4)
    )
}
theme_set(theme_pub())

save_pub <- function(plot, filename, width_mm = 183, height_mm = 120, dpi = 600) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4
  svglite::svglite(paste0(filename, ".svg"), width = width_in, height = height_in)
  print(plot)
  dev.off()
  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = width_in, height = height_in, family = "Arial")
  print(plot)
  dev.off()
  ragg::agg_tiff(paste0(filename, ".tiff"), width = width_in, height = height_in, units = "in", res = dpi)
  print(plot)
  dev.off()
  ragg::agg_png(paste0(filename, ".png"), width = width_in, height = height_in, units = "in", res = 220)
  print(plot)
  dev.off()
}

bootstrap_summary <- function(data, groups, value_col = "value", reps = 1500, seed = 20260715) {
  set.seed(seed)
  data %>%
    group_by(across(all_of(groups))) %>%
    group_modify(~ {
      x <- .x[[value_col]]
      x <- x[is.finite(x)]
      boots <- replicate(reps, mean(sample(x, length(x), replace = TRUE)))
      tibble(
        mean = mean(x),
        lower = unname(quantile(boots, 0.025)),
        upper = unname(quantile(boots, 0.975)),
        n = length(x)
      )
    }) %>%
    ungroup()
}

# Figure 1: closed-PDA agreement and numerical convergence
case_labels <- c(
  operational_20_30m = "Operational",
  low_pressure_stress_8_20m = "Stress"
)
entity <- read_csv(file.path(data_root, "hydraulic_validation", "pda_epanet_entity_comparisons.csv"), show_col_types = FALSE) %>%
  mutate(case_label = recode(case_class, !!!case_labels))
case_error <- read_csv(file.path(data_root, "hydraulic_validation", "pda_error_metrics_by_case.csv"), show_col_types = FALSE) %>%
  mutate(case_label = recode(case_class, !!!case_labels))
cases <- read_csv(file.path(data_root, "hydraulic_validation", "pda_validation_cases.csv"), show_col_types = FALSE) %>%
  mutate(case_label = recode(case_class, !!!case_labels))
write_csv(entity, file.path(source_dir, "Fig1_entity_comparisons.csv"))
write_csv(case_error, file.path(source_dir, "Fig1_case_errors.csv"))
write_csv(cases, file.path(source_dir, "Fig1_convergence_cases.csv"))

validation_colors <- c(
  "Operational" = pal[["blue"]],
  "Stress" = pal[["orange"]]
)
pressure_scatter <- entity %>% filter(metric == "active_node_pressure_m")
p1a <- ggplot(pressure_scatter, aes(reference, estimate, colour = case_label)) +
  geom_abline(slope = 1, intercept = 0, linewidth = 0.55, colour = pal[["ink"]], linetype = 2) +
  geom_point(size = 0.85, alpha = 0.50) +
  scale_colour_manual(values = validation_colors) +
  coord_equal() +
  labs(
    x = "EPANET pressure (m)", y = "Closed-PDA pressure (m)",
    title = "Hydraulic agreement", subtitle = "Identity line indicates exact agreement"
  ) +
  guides(colour = guide_legend(nrow = 1, override.aes = list(alpha = 1, size = 1.8))) +
  theme(legend.position = "top")

bland_altman <- pressure_scatter %>%
  mutate(mean_pressure_m = (reference + estimate) / 2, difference_m = estimate - reference)
ba_limits <- bland_altman %>%
  group_by(case_label) %>%
  summarise(
    bias = mean(difference_m),
    lower = bias - 1.96 * sd(difference_m),
    upper = bias + 1.96 * sd(difference_m),
    .groups = "drop"
  )
write_csv(bland_altman, file.path(source_dir, "Fig1_Bland_Altman_pressure.csv"))
p1b <- ggplot(bland_altman, aes(mean_pressure_m, difference_m, colour = case_label)) +
  geom_hline(data = ba_limits, aes(yintercept = bias), linewidth = 0.45) +
  geom_hline(data = ba_limits, aes(yintercept = lower), linewidth = 0.35, linetype = 2) +
  geom_hline(data = ba_limits, aes(yintercept = upper), linewidth = 0.35, linetype = 2) +
  geom_point(size = 0.72, alpha = 0.45) +
  facet_wrap(~case_label, scales = "free_y") +
  scale_colour_manual(values = validation_colors) +
  labs(x = "Mean pressure (m)", y = "Closed-PDA - EPANET (m)", title = "Bland-Altman consistency") +
  theme(legend.position = "none")

metric_labels <- c(
  active_node_pressure_m = "Node pressure",
  active_node_demand_m3s = "Emitter demand",
  active_pipe_flow_m3s = "Pipe flow",
  active_pipe_velocity_ms = "Pipe velocity",
  active_pipe_friction_loss_m = "Friction loss"
)
error_intervals <- case_error %>%
  mutate(
    metric_label = recode(metric, !!!metric_labels),
    mape_safe = pmax(mape_pct, 1e-8)
  ) %>%
  group_by(case_label, metric_label) %>%
  summarise(
    median = median(mape_safe),
    lower = quantile(mape_safe, 0.25),
    upper = quantile(mape_safe, 0.95),
    .groups = "drop"
  )
p1c <- ggplot(error_intervals, aes(median, metric_label, colour = case_label)) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = 0, linewidth = 0.65, position = position_dodge(width = 0.45)) +
  geom_point(size = 1.8, position = position_dodge(width = 0.45)) +
  scale_x_log10(labels = label_number(accuracy = 0.001)) +
  scale_colour_manual(values = validation_colors) +
  labs(
    x = "Absolute percentage error (median; 25th-95th percentile)", y = NULL,
    title = "Error across hydraulic quantities"
  ) +
  theme(legend.position = "top")

p1d <- ggplot(cases, aes(custom_iterations, colour = case_label)) +
  stat_ecdf(geom = "step", linewidth = 0.75) +
  geom_rug(aes(colour = case_label), sides = "b", alpha = 0.35, linewidth = 0.3) +
  scale_colour_manual(values = validation_colors) +
  scale_y_continuous(labels = label_percent(), limits = c(0, 1.02)) +
  labs(x = "Fixed-point iterations", y = "Cumulative fraction", title = "Convergence effort") +
  theme(legend.position = "none")

fig1 <- p1a + p1b + p1c + p1d +
  plot_layout(design = "AAB\nCCD", heights = c(1.05, 1)) +
  plot_annotation(tag_levels = "a")
save_pub(fig1, file.path(main_dir, "Figure1_closed_PDA_validation"), height_mm = 128)

# Figure 2: agricultural value under normal and disturbed operation
normal <- read_csv(file.path(data_root, "paper_statistics", "normal_scenario_aggregates.csv"), show_col_types = FALSE) %>%
  mutate(comparison_policy = factor(
    comparison_policy,
    levels = c("Original near-to-far", "Random", "Tuned NSGA-II")
  ))
fault <- read_csv(file.path(data_root, "paper_statistics", "fault_scenario_aggregates.csv"), show_col_types = FALSE) %>%
  mutate(
    experiment = recode(experiment, single_fault = "Single fault/recovery", multi_fault = "Multiple faults/recoveries"),
    algorithm = factor(algorithm, levels = c("nsga2", "spea2", "rvea", "moead"))
  )
write_csv(normal, file.path(source_dir, "Fig2_normal_scenarios.csv"))
write_csv(fault, file.path(source_dir, "Fig2_fault_scenarios.csv"))

normal_long <- normal %>%
  select(scenario_id, comparison_policy, achievement_pct, pressure_uniformity_index_pct, feasible_group_ratio_pct) %>%
  pivot_longer(-c(scenario_id, comparison_policy), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(
    metric,
    achievement_pct = "Irrigation achievement (%)",
    pressure_uniformity_index_pct = "Pressure uniformity (%)",
    feasible_group_ratio_pct = "Feasible groups (%)"
  ))
normal_ci <- bootstrap_summary(normal_long, c("comparison_policy", "metric"))
normal_colors <- c(
  "Original near-to-far" = pal[["neutral"]],
  "Random" = pal[["orange"]],
  "Tuned NSGA-II" = pal[["blue"]]
)
p2a <- ggplot(normal_ci, aes(mean, comparison_policy, colour = comparison_policy)) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = 0, linewidth = 0.85) +
  geom_point(size = 2.5) +
  facet_wrap(~metric, scales = "free_x", nrow = 1) +
  scale_colour_manual(values = normal_colors) +
  labs(
    x = "Scenario mean and 95% bootstrap CI", y = NULL,
    title = "Normal-operation performance across 100 pressure scenarios"
  ) +
  theme(legend.position = "none", panel.grid.major.y = element_line(colour = "#E5E7E9"))

fault_wide <- fault %>%
  select(
    experiment, scenario_id, algorithm, decision_mode,
    achievement_pct, affected_target_met_pct, average_deficit_m3_per_emitter
  ) %>%
  pivot_wider(names_from = decision_mode, values_from = c(
    achievement_pct, affected_target_met_pct, average_deficit_m3_per_emitter
  )) %>%
  transmute(
    experiment, scenario_id, algorithm,
    `Achievement gain` = achievement_pct_Dynamic - achievement_pct_Static,
    `Affected recovery` = affected_target_met_pct_Dynamic - affected_target_met_pct_Static,
    `Deficit reduction` = average_deficit_m3_per_emitter_Static - average_deficit_m3_per_emitter_Dynamic
  ) %>%
  pivot_longer(-c(experiment, scenario_id, algorithm), names_to = "metric", values_to = "benefit") %>%
  mutate(metric = factor(metric, levels = c("Achievement gain", "Affected recovery", "Deficit reduction")))
write_csv(fault_wide, file.path(source_dir, "Fig2_dynamic_minus_static_effects.csv"))

p2b <- ggplot(fault_wide, aes(algorithm, benefit, fill = algorithm)) +
  geom_hline(yintercept = 0, colour = pal[["neutral"]], linewidth = 0.4, linetype = 2) +
  geom_violin(scale = "width", trim = TRUE, alpha = 0.58, linewidth = 0.3) +
  stat_summary(fun = median, geom = "point", shape = 21, size = 1.5, fill = "white", colour = pal[["ink"]]) +
  facet_grid(metric ~ experiment, scales = "free_y", switch = "y") +
  scale_fill_manual(values = algorithm_colors) +
  labs(
    x = NULL, y = "Dynamic reoptimization benefit",
    title = "Paired scenario-level improvement over static scheduling"
  ) +
  theme(
    legend.position = "none", panel.grid.major.y = element_line(colour = "#E8EAEC"),
    strip.placement = "outside", strip.text.y.left = element_text(angle = 0, size = 6.2)
  )

effect_map <- fault_wide %>%
  group_by(experiment, algorithm, metric) %>%
  summarise(effect = median(benefit), .groups = "drop") %>%
  group_by(experiment, metric) %>%
  mutate(relative_effect = effect / max(abs(effect))) %>%
  ungroup() %>%
  mutate(experiment_short = recode(experiment, "Multiple faults/recoveries" = "Multiple", "Single fault/recovery" = "Single"))
p2c <- ggplot(effect_map, aes(experiment_short, algorithm, fill = relative_effect)) +
  geom_tile(colour = "white", linewidth = 0.7) +
  geom_text(aes(label = number(effect, accuracy = 0.1)), size = 2.15, colour = pal[["ink"]]) +
  facet_wrap(~metric, ncol = 1) +
  scale_fill_gradient2(low = "#F2F3F4", mid = "#BFD5E5", high = pal[["blue"]], midpoint = 0, limits = c(-1, 1)) +
  labs(
    x = NULL, y = NULL,
    title = "Median recovery-effect matrix",
    subtitle = "Values: percentage points (top two) or m3/emitter (bottom)"
  ) +
  theme(
    legend.position = "none", axis.text.x = element_text(angle = 22, hjust = 1),
    panel.grid = element_blank(), axis.line = element_blank(), axis.ticks = element_blank()
  )

fig2 <- p2a / (p2b | p2c) +
  plot_layout(heights = c(0.76, 1.45), widths = c(1.75, 1)) +
  plot_annotation(tag_levels = "a")
save_pub(fig2, file.path(main_dir, "Figure2_agricultural_operation"), height_mm = 142)

# Figure 3: four-dimensional Pareto quality and convergence
quality <- read_csv(file.path(data_root, "algorithm_quality", "pareto_quality_metrics.csv"), show_col_types = FALSE) %>%
  mutate(algorithm = factor(algorithm, levels = c("nsga2", "spea2", "rvea", "moead")))
convergence <- read_csv(file.path(source_dir, "algorithm_convergence_summary.csv"), show_col_types = FALSE) %>%
  mutate(algorithm = factor(algorithm, levels = levels(quality$algorithm)))
fronts <- read_csv(file.path(source_dir, "Fig3_heldout_pareto_front_points.csv"), show_col_types = FALSE) %>%
  mutate(algorithm = factor(algorithm, levels = levels(quality$algorithm)))
write_csv(quality, file.path(source_dir, "Fig3_Pareto_quality_runs.csv"))

context_scores <- quality %>%
  group_by(context_id, algorithm) %>%
  summarise(mean_hv = mean(hv_4d), .groups = "drop") %>%
  group_by(context_id) %>%
  summarise(hv_range = max(mean_hv) - min(mean_hv), .groups = "drop")
representative_context <- context_scores$context_id[which.max(context_scores$hv_range)]
representative_front <- fronts %>%
  filter(context_id == representative_context) %>%
  mutate(point_key = interaction(algorithm, replicate, point_id, drop = TRUE))
write_csv(representative_front, file.path(source_dir, "Fig3_representative_Pareto_front.csv"))
representative_front_plot <- representative_front %>%
  distinct(algorithm, pressure_nonuniformity, hydraulic_cv, .keep_all = TRUE)

front_long <- representative_front %>%
  select(point_key, algorithm, achievement_loss, average_deficit_m3_per_emitter, pressure_nonuniformity, hydraulic_cv) %>%
  pivot_longer(-c(point_key, algorithm), names_to = "objective", values_to = "value") %>%
  group_by(objective) %>%
  mutate(normalized = ifelse(max(value) > min(value), (value - min(value)) / (max(value) - min(value)), 0)) %>%
  ungroup() %>%
  mutate(objective = factor(
    objective,
    levels = c("achievement_loss", "average_deficit_m3_per_emitter", "pressure_nonuniformity", "hydraulic_cv"),
    labels = c("Achievement\nloss", "Mean\ndeficit", "Pressure\nnonuniformity", "Hydraulic\nviolation")
  ))
front_medians <- front_long %>%
  group_by(algorithm, objective) %>%
  summarise(normalized = median(normalized), .groups = "drop")
p3a <- ggplot(
  representative_front_plot,
  aes(pressure_nonuniformity, hydraulic_cv, colour = algorithm)
) +
  geom_point(alpha = 0.34, size = 0.9, stroke = 0) +
  facet_wrap(~algorithm, nrow = 1) +
  scale_colour_manual(values = algorithm_colors) +
  scale_x_continuous(breaks = c(2.5, 3.0), labels = label_number(accuracy = 0.1)) +
  labs(
    x = "Pressure nonuniformity", y = "Hydraulic violation",
    title = "Representative Pareto-front projection",
    subtitle = paste0("Held-out context ", representative_context, "; pooled nondominated solutions from 20 runs")
  ) +
  guides(colour = "none") +
  theme(legend.position = "top", panel.grid.major.x = element_line(colour = "#E3E6E8"))

quality_rank_long <- quality %>%
  select(context_id, replicate, algorithm, hv_4d, igd_plus_4d, additive_epsilon_4d, spacing_4d, elapsed_s) %>%
  pivot_longer(-c(context_id, replicate, algorithm), names_to = "metric", values_to = "value") %>%
  group_by(context_id, replicate, metric) %>%
  mutate(rank = rank(ifelse(metric == "hv_4d", -value, value), ties.method = "average")) %>%
  ungroup() %>%
  mutate(metric = recode(
    metric, hv_4d = "HV", igd_plus_4d = "IGD+",
    additive_epsilon_4d = "Additive epsilon", spacing_4d = "Spacing", elapsed_s = "Wall time"
  ), metric = factor(metric, levels = c("HV", "IGD+", "Additive epsilon", "Spacing", "Wall time")))
rank_map <- quality_rank_long %>%
  group_by(algorithm, metric) %>%
  summarise(mean_rank = mean(rank), .groups = "drop")
p3b <- ggplot(rank_map, aes(metric, algorithm, fill = mean_rank)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = number(mean_rank, accuracy = 0.01)), size = 2.25) +
  scale_fill_gradient(low = "#E4F1EC", high = pal[["orange"]], limits = c(1, 4), name = "Mean rank") +
  labs(x = NULL, y = NULL, title = "Cross-context performance ranks", subtitle = "Rank 1 is best") +
  theme(
    legend.position = "top", axis.text.x = element_text(angle = 24, hjust = 1),
    panel.grid = element_blank(), axis.line = element_blank(), axis.ticks = element_blank()
  )

p3c <- ggplot(convergence, aes(function_evaluations, hv_mean, colour = algorithm, fill = algorithm)) +
  geom_ribbon(aes(ymin = hv_q025, ymax = hv_q975), alpha = 0.11, colour = NA) +
  geom_line(linewidth = 0.72) +
  scale_colour_manual(values = algorithm_colors) +
  scale_fill_manual(values = algorithm_colors) +
  labs(
    x = "Function evaluations", y = "Mean 4D hypervolume",
    title = "Equal-budget convergence", subtitle = "Ribbon: 2.5th-97.5th percentile"
  ) +
  theme(legend.position = "none")

quality_context <- quality %>%
  group_by(context_id, algorithm) %>%
  summarise(
    elapsed_s = mean(elapsed_s), hv_4d = mean(hv_4d),
    nondominated_count = mean(nondominated_count), .groups = "drop"
  )
quality_centres <- quality_context %>%
  group_by(algorithm) %>%
  summarise(
    elapsed_s = mean(elapsed_s), hv_4d = mean(hv_4d),
    nondominated_count = mean(nondominated_count), .groups = "drop"
  )
p3d <- ggplot(quality_context, aes(elapsed_s, hv_4d, colour = algorithm, size = nondominated_count)) +
  geom_point(alpha = 0.20) +
  geom_point(data = quality_centres, shape = 21, fill = "white", stroke = 1, alpha = 1) +
  geom_text_repel(
    data = quality_centres, aes(label = toupper(as.character(algorithm))),
    size = 2.2, show.legend = FALSE, box.padding = 0.25, min.segment.length = 0
  ) +
  scale_colour_manual(values = algorithm_colors) +
  scale_size_continuous(range = c(0.8, 3.4)) +
  labs(
    x = "Mean wall time per run (s)", y = "Mean 4D hypervolume",
    title = "Quality-cost frontier", subtitle = "Points are held-out contexts; size is front size"
  ) +
  theme(legend.position = "none")

fig3 <- (p3a | p3b) / (p3c | p3d) +
  plot_layout(widths = c(1.45, 1), heights = c(1.1, 1)) +
  plot_annotation(tag_levels = "a")
save_pub(fig3, file.path(main_dir, "Figure3_multiobjective_algorithm_quality"), height_mm = 132)

# Figure 4: cache-layer attribution, memory trade-off and scaling
cache <- read_csv(file.path(data_root, "cache_scalability", "cache_ablation_runs.csv"), show_col_types = FALSE) %>%
  mutate(
    cache_level_num = cache_level,
    cache_level = factor(cache_level, levels = 0:3, labels = c("None", "L1 pipe", "L1+L2 group", "L1+L2+L3 plan")),
    cache_state = recode(cache_state, cold = "Cold", warm_cross_seed = "Cross-seed warm", hot_exact_replay = "Exact hot replay")
  )
scale_data <- read_csv(file.path(data_root, "cache_scalability", "network_scalability_runs.csv"), show_col_types = FALSE) %>%
  mutate(cache_level = factor(cache_level, levels = 0:3, labels = c("None", "L1", "L1+L2", "L1+L2+L3")))
write_csv(cache, file.path(source_dir, "Fig4_cache_ablation_runs.csv"))
write_csv(scale_data, file.path(source_dir, "Fig4_scalability_runs.csv"))

level_colors <- c(
  "None" = pal[["neutral"]], "L1 pipe" = pal[["orange"]],
  "L1+L2 group" = pal[["teal"]], "L1+L2+L3 plan" = pal[["blue"]]
)
cold_times <- cache %>%
  filter(cache_state == "Cold") %>%
  group_by(cache_level_num) %>%
  summarise(elapsed_s = median(elapsed_s), .groups = "drop") %>%
  arrange(cache_level_num)
t <- cold_times$elapsed_s
waterfall <- tibble(
  x = 1:5,
  label = c("No cache", "L1 saved", "L2 saved", "L3 saved", "Final"),
  start = c(0, t[1], t[2], t[3], 0),
  end = c(t[1], t[2], t[3], t[4], t[4]),
  kind = c("Baseline", "Saved", "Saved", "Saved", "Final"),
  annotation = c(
    paste0(number(t[1], accuracy = 0.1), " s"),
    paste0("-", number(t[1] - t[2], accuracy = 0.1), " s"),
    paste0("-", number(t[2] - t[3], accuracy = 0.1), " s"),
    paste0("-", number(t[3] - t[4], accuracy = 0.01), " s"),
    paste0(number(t[4], accuracy = 0.01), " s")
  )
)
p4a <- ggplot(waterfall, aes(x = x)) +
  geom_rect(
    aes(xmin = x - 0.34, xmax = x + 0.34, ymin = pmin(start, end), ymax = pmax(start, end), fill = kind),
    linewidth = 0
  ) +
  geom_segment(
    data = waterfall[1:4, ], aes(x = x + 0.34, xend = x + 1 - 0.34, y = end, yend = end),
    colour = pal[["light"]], linewidth = 0.45, linetype = 2
  ) +
  geom_text(aes(y = pmax(start, end) + max(t) * 0.045, label = annotation), size = 2.15) +
  scale_x_continuous(breaks = waterfall$x, labels = waterfall$label) +
  scale_fill_manual(values = c(Baseline = pal[["neutral"]], Saved = pal[["teal"]], Final = pal[["blue"]])) +
  expand_limits(y = max(t) * 1.14) +
  labs(x = NULL, y = "Median cold-cache runtime (s)", title = "Incremental time saved by each cache layer") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1))

cache_efficiency <- cache %>%
  group_by(cache_state, cache_level, cache_level_num) %>%
  summarise(
    elapsed_s = median(elapsed_s),
    cache_mb = median(cache_estimated_bytes) / 1024^2,
    top_hit_rate = case_when(
      first(cache_level_num) == 0 ~ 0,
      first(cache_level_num) == 1 ~ median(pipe_cache_hit_rate_pct),
      first(cache_level_num) == 2 ~ median(group_cache_hit_rate_pct),
      TRUE ~ median(plan_cache_hit_rate_pct)
    ),
    .groups = "drop"
  ) %>%
  mutate(state_short = factor(
    recode(cache_state, "Cold" = "Cold", "Cross-seed warm" = "Warm", "Exact hot replay" = "Hot"),
    levels = c("Cold", "Warm", "Hot")
  ))
p4b <- ggplot(cache_efficiency, aes(cache_mb + 0.005, elapsed_s, colour = cache_level, shape = state_short, size = top_hit_rate + 2)) +
  geom_point(alpha = 0.88) +
  scale_x_log10(labels = label_number(accuracy = 0.01)) +
  scale_y_log10(labels = label_number(accuracy = 0.01)) +
  scale_colour_manual(values = level_colors) +
  scale_size_continuous(range = c(1.5, 5.5)) +
  labs(
    x = "Cache storage (MB, log scale)", y = "Median runtime (s, log scale)",
    title = "Runtime-memory efficiency frontier", subtitle = "Point size: hit rate of the highest enabled layer"
  ) +
  guides(
    size = "none",
    colour = guide_legend(nrow = 1, order = 1),
    shape = guide_legend(nrow = 1, order = 2)
  ) +
  theme(
    legend.position = "top", legend.box = "vertical",
    legend.text = element_text(size = 5.5), legend.key.width = grid::unit(7, "pt")
  )

hit_map <- cache_efficiency %>%
  mutate(cache_level = factor(cache_level, levels = rev(levels(cache$cache_level))))
p4c <- ggplot(hit_map, aes(state_short, cache_level, fill = top_hit_rate)) +
  geom_tile(colour = "white", linewidth = 0.75) +
  geom_text(aes(label = paste0(number(top_hit_rate, accuracy = 0.1), "%")), size = 2.15) +
  scale_fill_gradient(low = "#F1F3F4", high = pal[["teal"]], limits = c(0, 100)) +
  labs(x = NULL, y = NULL, title = "Highest-layer cache reuse") +
  theme(
    legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1),
    panel.grid = element_blank(), axis.line = element_blank(), axis.ticks = element_blank()
  )

p4d <- scale_data %>%
  group_by(emitter_count, cache_level) %>%
  summarise(elapsed_s = median(elapsed_s), .groups = "drop") %>%
  ggplot(aes(emitter_count, elapsed_s, colour = cache_level)) +
  geom_line(linewidth = 0.72) +
  geom_point(size = 1.7) +
  scale_x_log10(breaks = c(60, 120, 240, 480)) +
  scale_y_log10(labels = label_number(accuracy = 0.01)) +
  scale_colour_manual(values = c(
    "None" = pal[["neutral"]], "L1" = pal[["orange"]],
    "L1+L2" = pal[["teal"]], "L1+L2+L3" = pal[["blue"]]
  )) +
  labs(x = "Emitters", y = "Median evaluation time (s)", title = "Measured network-size scaling") +
  guides(colour = guide_legend(nrow = 2)) +
  theme(legend.position = "top")

fig4 <- ((p4a + labs(tag = "a") + theme(plot.tag.position = c(0, 1))) | (p4b + labs(tag = "b"))) /
  ((p4c + labs(tag = "c")) | (p4d + labs(tag = "d"))) +
  plot_layout(widths = c(1.08, 1.35), heights = c(1.05, 1))
save_pub(fig4, file.path(main_dir, "Figure4_cache_ablation_scalability"), height_mm = 132)

# Figure 5: information value, deployability and computational burden
info <- read_csv(file.path(data_root, "information_set", "information_set_policy_metrics.csv"), show_col_types = FALSE) %>%
  mutate(
    experiment = recode(experiment, single_fault = "Single fault/recovery", multi_fault = "Multiple faults/recoveries"),
    policy = factor(
      policy,
      levels = c("oracle_event_driven", "causal_mean_event_driven", "causal_persistence_event_driven", "causal_mean_rolling_group"),
      labels = c("Oracle head", "Causal mean", "Causal persistence", "Causal rolling")
    )
  )
write_csv(info, file.path(source_dir, "Fig5_information_set_scenarios.csv"))
info_colors <- c(
  "Oracle head" = pal[["purple"]], "Causal mean" = pal[["blue"]],
  "Causal persistence" = pal[["orange"]], "Causal rolling" = pal[["teal"]]
)
info_centres <- info %>%
  group_by(experiment, policy) %>%
  summarise(
    total_optimization_time_s = median(total_optimization_time_s),
    achievement_pct = median(achievement_pct), .groups = "drop"
  )
p5a <- ggplot(info, aes(total_optimization_time_s, achievement_pct, colour = policy)) +
  geom_point(alpha = 0.32, size = 1.05) +
  geom_point(data = info_centres, shape = 21, fill = "white", stroke = 1, size = 2.7) +
  geom_text_repel(
    data = info_centres, aes(label = policy), size = 2.15,
    show.legend = FALSE, box.padding = 0.25, min.segment.length = 0
  ) +
  facet_wrap(~experiment, scales = "free_y") +
  scale_x_log10(labels = label_number(accuracy = 1)) +
  scale_colour_manual(values = info_colors) +
  labs(
    x = "Total optimization time (s, log scale)", y = "Irrigation achievement (%)",
    title = "Operational value versus computational burden",
    subtitle = "Open symbols mark policy medians"
  ) +
  theme(legend.position = "none")

info_long <- info %>%
  select(
    experiment, scenario_id, policy, achievement_pct,
    average_deficit_m3_per_emitter, pressure_uniformity_index_pct,
    total_optimization_time_s
  ) %>%
  pivot_longer(-c(experiment, scenario_id, policy), names_to = "metric", values_to = "value") %>%
  mutate(
    metric_label = recode(
      metric,
      achievement_pct = "Achievement",
      average_deficit_m3_per_emitter = "Deficit reduction",
      pressure_uniformity_index_pct = "Pressure uniformity",
      total_optimization_time_s = "Time saving"
    ),
    experiment_short = recode(experiment, "Multiple faults/recoveries" = "Multiple", "Single fault/recovery" = "Single")
  )
info_baseline <- info_long %>%
  filter(policy == "Causal mean") %>%
  select(experiment, scenario_id, metric, baseline = value)
info_effects <- info_long %>%
  filter(policy != "Causal mean") %>%
  left_join(info_baseline, by = c("experiment", "scenario_id", "metric")) %>%
  mutate(value = ifelse(
    metric %in% c("average_deficit_m3_per_emitter", "total_optimization_time_s"),
    baseline - value, value - baseline
  ))
write_csv(info_effects, file.path(source_dir, "Fig5_paired_effects_vs_causal_mean.csv"))
info_effect_ci <- bootstrap_summary(info_effects, c("experiment_short", "policy", "metric_label"))
p5b <- ggplot(info_effect_ci, aes(mean, policy, colour = policy)) +
  geom_vline(xintercept = 0, colour = pal[["neutral"]], linewidth = 0.4, linetype = 2) +
  geom_errorbar(aes(xmin = lower, xmax = upper), orientation = "y", width = 0, linewidth = 0.85) +
  geom_point(size = 2.25) +
  facet_grid(metric_label ~ experiment_short, scales = "free_x", switch = "y") +
  scale_colour_manual(values = info_colors) +
  labs(
    x = "Paired directional effect vs causal mean (95% scenario bootstrap CI)", y = NULL,
    title = "Value of additional information or reoptimization"
  ) +
  theme(
    legend.position = "none", panel.grid.major.y = element_line(colour = "#E8EAEC"),
    strip.placement = "outside", strip.text.y.left = element_text(angle = 0, size = 6.0)
  )

best_rates <- info_long %>%
  mutate(score = ifelse(metric %in% c("average_deficit_m3_per_emitter", "total_optimization_time_s"), -value, value)) %>%
  group_by(experiment_short, scenario_id, metric_label) %>%
  mutate(best = score == max(score)) %>%
  ungroup() %>%
  group_by(experiment_short, policy, metric_label) %>%
  summarise(best_scenario_pct = mean(best) * 100, .groups = "drop")
p5c <- ggplot(best_rates, aes(best_scenario_pct, policy, colour = policy)) +
  geom_segment(aes(x = 0, xend = best_scenario_pct, yend = policy), colour = pal[["light"]], linewidth = 1.1) +
  geom_point(size = 2.2) +
  geom_text(
    aes(
      label = paste0(number(best_scenario_pct, accuracy = 1), "%"),
      hjust = ifelse(best_scenario_pct >= 85, 1.12, -0.12)
    ),
    size = 2.0, show.legend = FALSE
  ) +
  facet_grid(metric_label ~ experiment_short, switch = "y") +
  scale_x_continuous(limits = c(0, 105), breaks = c(0, 50, 100)) +
  scale_colour_manual(values = info_colors) +
  labs(x = "Scenarios in which policy is best (%)", y = NULL, title = "Scenario-level policy wins") +
  theme(
    legend.position = "none", panel.grid.major.y = element_blank(),
    strip.placement = "outside", strip.text.y.left = element_text(angle = 0, size = 6.0)
  )

fig5 <- p5a / (p5b | p5c) +
  plot_layout(heights = c(0.92, 1.55), widths = c(1.35, 1)) +
  plot_annotation(tag_levels = "a")
save_pub(fig5, file.path(main_dir, "Figure5_information_set"), height_mm = 146)

# Supplementary Figure 1: surrogate predictive generalization
surrogate_metrics <- read_csv(file.path(data_root, "surrogate_grouped", "surrogate_test_metrics.csv"), show_col_types = FALSE) %>%
  filter(topology_id == "all", target != "feasibility_classification") %>%
  mutate(
    split = recode(split, test = "Internal grouped test", external_test = "External topologies"),
    target = recode(
      target,
      obj_achievement_loss = "Achievement loss", obj_average_deficit = "Mean deficit",
      obj_pressure_nonuniformity = "Pressure nonuniformity", constraint_violation = "Constraint violation"
    )
  )
feasibility <- read_csv(file.path(data_root, "surrogate_grouped", "surrogate_test_metrics.csv"), show_col_types = FALSE) %>%
  filter(topology_id == "all", target == "feasibility_classification") %>%
  mutate(split = recode(split, test = "Internal grouped test", external_test = "External topologies"))
write_csv(surrogate_metrics, file.path(source_dir, "FigS1_surrogate_regression_metrics.csv"))
write_csv(feasibility, file.path(source_dir, "FigS1_surrogate_feasibility_metrics.csv"))
model_colors <- c(catboost = pal[["orange"]], lightgbm = pal[["teal"]], xgboost = pal[["blue"]])
ps1a <- ggplot(surrogate_metrics, aes(target, nrmse_iqr, fill = model_family)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  facet_wrap(~split, scales = "free_y") +
  scale_fill_manual(values = model_colors) +
  labs(x = NULL, y = "IQR-normalized RMSE", title = "Objective prediction") +
  theme(legend.position = "top", axis.text.x = element_text(angle = 25, hjust = 1))
ps1b <- ggplot(feasibility, aes(model_family, balanced_accuracy, fill = model_family)) +
  geom_col(width = 0.65) +
  facet_wrap(~split) +
  scale_fill_manual(values = model_colors) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = NULL, y = "Balanced accuracy", title = "Feasibility classification") +
  theme(legend.position = "none")
figs1 <- ps1a / ps1b + plot_layout(heights = c(1.25, 1)) + plot_annotation(tag_levels = "a")
save_pub(figs1, file.path(supp_dir, "FigureS1_surrogate_generalization"), height_mm = 112)

# Supplementary Figure 2: surrogate equivalence and runtime gate
equivalence <- read_csv(file.path(data_root, "paper_tables", "supplementary", "TableS9_surrogate_equivalence_gate.csv"), show_col_types = FALSE) %>%
  mutate(
    experiment = recode(experiment, normal = "Normal", single_fault = "Single fault", multi_fault = "Multiple faults"),
    metric = recode(
      metric,
      achievement_pct = "Achievement", unmet_emitter_ratio_pct = "Unmet ratio",
      average_deficit_m3_per_emitter = "Mean deficit", max_deficit_m3 = "Maximum deficit",
      pressure_uniformity_index_pct = "Pressure uniformity", feasible_group_ratio_pct = "Feasible groups"
    )
  )
runtime <- read_csv(file.path(data_root, "paper_tables", "supplementary", "TableS10_surrogate_runtime_exact_evaluations.csv"), show_col_types = FALSE)
write_csv(equivalence, file.path(source_dir, "FigS2_surrogate_equivalence.csv"))
write_csv(runtime, file.path(source_dir, "FigS2_surrogate_runtime.csv"))
ps2a <- ggplot(equivalence, aes(metric, maximum_normalized_difference_pct, fill = experiment)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  geom_hline(yintercept = 5, colour = pal[["red"]], linetype = 2, linewidth = 0.5) +
  scale_fill_manual(values = c("Normal" = pal[["blue"]], "Single fault" = pal[["orange"]], "Multiple faults" = pal[["teal"]])) +
  labs(x = NULL, y = "Maximum normalized difference (%)", title = "Predefined 5% equivalence gate") +
  theme(legend.position = "top", axis.text.x = element_text(angle = 30, hjust = 1))
runtime_long <- runtime %>%
  select(method, normal_mean_optimization_time_s, normal_mean_exact_evaluations) %>%
  pivot_longer(-method, names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(
    metric,
    normal_mean_optimization_time_s = "Optimization time (s)",
    normal_mean_exact_evaluations = "Exact evaluations"
  ))
ps2b <- ggplot(runtime_long, aes(method, value, fill = method)) +
  geom_col(width = 0.62) +
  facet_wrap(~metric, scales = "free_y") +
  scale_fill_manual(values = c("Exact NSGA-II" = pal[["blue"]], "Surrogate-assisted NSGA-II" = pal[["orange"]])) +
  labs(x = NULL, y = NULL, title = "Evaluation count does not predict wall time") +
  theme(legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1))
figs2 <- ps2a / ps2b + plot_annotation(tag_levels = "a")
save_pub(figs2, file.path(supp_dir, "FigureS2_surrogate_equivalence_runtime"), height_mm = 108)

cat("Regenerated and overwrote 5 main figures; supplementary figures refreshed.\n")
