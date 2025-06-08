# Causal AI Analysis Application

A Streamlit-based application for causal discovery and causal inference analysis.

## TODOs

### Development Workflow
- [ ] Create **develop branch** so we don't need to change the hosted one
  - Create **branching strategy**
- [ ] Fix errors in **regression tests** if there are still any
  - Add **automatic email sending** using Yahoo

### Research Tasks
- [ ] Check how **DoWhy** uses confounder variables and the graph, and how we pass them
- [ ] Use **2 different causal discovery algorithms** and find a way to merge them together

### Bug Fixes
- [ ] Fix **DirectLiNGAM constraint handling**
  - **Issue:** When required edges are specified, DirectLiNGAM only returns those edges instead of the full causal graph
  - **Solution:** Add integration tests for DirectLiNGAM to ensure proper constraint handling and edge discovery
