def tpuv4_linear_model(cycles, s_row=1, s_col=1, t_time=1):
    """
    TPUv4 linear model for converting cycles to time in microseconds.
    
    Args:
        cycles: Total compute cycles
        s_row: Spatial dimension rows
        s_col: Spatial dimension columns
        t_time: Temporal dimension
    
    Returns:
        Time in microseconds
    """
    if s_row <=128 and s_col <=128 and t_time <=128:
        return 0.002762 * cycles - 0.062665
    elif s_row <=1024 and s_col <=1024 and t_time <=1024:
        return 0.000388 * cycles + 2.05942
    else:
        return 0.000202 * cycles + 29.7217
def tpuv5e_linear_model(cycles, s_row=1, s_col=1, t_time=1):
    """
    TPUv5e linear model for converting cycles to time in microseconds.
    
    Args:
        cycles: Total compute cycles
        s_row: Spatial dimension rows
        s_col: Spatial dimension columns
        t_time: Temporal dimension
    
    Returns:
        Time in microseconds
    """
    # TODO: Modify for V5
    if s_row <=128 and s_col <=128 and t_time <=128:
        return  0.002133 * cycles - 0.168796
    elif s_row <=1024 and s_col <=1024 and t_time <=1024:
        return 0.000167 * cycles + 1.158923
    else:
        return 0.000159 * cycles -0.380696

def tpuv6e_linear_model(cycles, s_row=1, s_col=1, t_time=1):
    """
    TPUv6e linear model for converting cycles to time in microseconds.
    
    Args:
        cycles: Total compute cycles
        s_row: Spatial dimension rows
        s_col: Spatial dimension columns
        t_time: Temporal dimension
    
    Returns:
        Time in microseconds
    """
    if s_row <=128 and s_col <=128 and t_time <=128:
        return 0.001389 * cycles + 0.604798
    elif s_row <=1024 and s_col <=1024 and t_time <=1024:
        return 0.000068 * cycles + 1.546793
    else:
        return 0.000040 * cycles + 4.384712
    