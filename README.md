# Causal AI Analysis Application

A Streamlit-based application for causal discovery and causal inference analysis.

## TODOs

### Development Workflow
- [ ] Create **develop branch** so we don't need to change the hosted one
  - Create **branching strategy**
- [ ] Fix errors in **regression tests** if there are still any
  - Add **automatic email sending** using Yahoo

### Research and Development Tasks
- [ ] Check how **DoWhy** uses confounder variables and the graph, and how we pass them
  - Right now, we can run causal inference without the step of causal discovery (causal graph) and also we can skip passing confounders to DoWhy
- [ ] Use **2 different causal discovery algorithms** and find a way to merge them together
  - **Idea:** Take edges from both graphs except forbidden edges. This ensures we don't use edges that make no sense based on domain knowledge, but stay open for new ideas.
  - **Rationale:** Currently we select treatment and outcome, and more edges can be beneficial because we control for confounders anyway.

### Bug Fixes
- [ ] Fix **DirectLiNGAM constraint handling**
  - **Issue:** When required edges are specified, DirectLiNGAM only returns those edges instead of the full causal graph
  - **Solution:** Add integration tests for DirectLiNGAM to ensure proper constraint handling and edge discovery
