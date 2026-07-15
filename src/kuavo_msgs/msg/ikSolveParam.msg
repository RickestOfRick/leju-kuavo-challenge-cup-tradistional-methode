# snopt params
float64 major_optimality_tol
float64 major_feasibility_tol
float64 minor_feasibility_tol
float64 major_iterations_limit
# constraint and cost params
float64 oritation_constraint_tol
float64 pos_constraint_tol # work when pos_cost_weight > 0.0
float64 pos_cost_weight
uint8 constraint_mode   # 2-bit constraint mode:  bit1: position hard(1)/soft(0) bit0: orientation hard(1)/soft(0) Mapping (binary -> decimal): 00->0, 11->3, 01->1, 10->2

# Optional right-arm target used by task-level strict multi-reference IK.
# When enabled, the solver targets zarm_r6_link at its current pose plus the
# requested world-frame XYZ offset and locks zarm_r7_joint to right_q7_lock_value.
bool use_right_r6_offset
bool lock_right_q7
float64 right_q7_lock_value
float64[3] right_r6_offset
