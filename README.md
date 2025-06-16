# Causal AI Analysis Application

A Streamlit-based application for causal discovery and causal inference analysis.

## TODOs

### Development Workflow
- [x] Create **develop branch** so we don't need to change the hosted one
  - Create **branching strategy**
- [x] Fix errors in **regression tests** if there are still any
  - Add **automatic email sending** using Yahoo

### Research and Development Tasks
- [x] Check **DirectLiNGAM constraints** implementation and validation
- [x] Check how **DoWhy** uses confounder variables and the graph, and how we pass them
  - Right now, we can run causal inference without the step of causal discovery (causal graph) and also we can skip passing confounders to DoWhy
- [ ] **Manual Constraint Editing Interface**
  - Implement interactive constraint editor allowing operators to review and modify AI-generated constraints before causal discovery
  - Support fully manual constraint specification when AI assistance is not used or unavailable
  - Provide validation and conflict resolution for user-defined constraints
- [ ] **Algorithm Selection Interface**
  - Implement user interface for configuring and selecting causal discovery algorithms and estimation methods
  - Support for multiple algorithm selection and ensemble approaches in future iterations
- [ ] Use **2 different causal discovery algorithms** and find a way to merge them together
  - **Idea:** Take edges from both graphs except forbidden edges. This ensures we don't use edges that make no sense based on domain knowledge, but stay open for new ideas.
  - **Rationale:** Currently we select treatment and outcome, and more edges can be beneficial because we control for confounders anyway.

### Bug Fixes
- [x] Fix **DirectLiNGAM constraint handling**
  - **Issue:** When required edges are specified, DirectLiNGAM only returns those edges instead of the full causal graph
  - **Solution:** Add integration tests for DirectLiNGAM to ensure proper constraint handling and edge discovery
