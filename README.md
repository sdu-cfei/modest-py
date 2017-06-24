# FMI-compliant Model Estimation in Python

![modestpy](/docs/img/modest-logo.png)

**modestpy** facilitates parameter estimation in models compliant with [Functional Mock-up Interface](https://fmi-standard.org/).


### Highlights

1) Estimate parameters using combined genetic algorithm (GA) and pattern search (PS),
  and select multiple random learning periods to avoid overfitting (dots represent switch from GA to PS):
![Error-evolution](/docs/img/err_evo.png)

2) Inspect visually GA evolution to ensure global search:
![GA-evolution](/docs/img/ga_evolution.png)

3) Analyze interdependencies between parameters:
![Intedependencies](/docs/img/all_estimates.png)


