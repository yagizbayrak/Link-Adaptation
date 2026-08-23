// Learned link adaptation: the inner loop choice shifted by the offset the trained network selects and clipped to the supported scheme range.
module dqn_la (
    input logic clock,
    input logic reset,
    input logic start,
    input logic signed [31:0] sinr_q16,
    input logic signed [831:0] observation,
    output logic [4:0] mcs_index,
    output logic done
);

logic [4:0] base_index;
logic [2:0] action;
logic signed [5:0] offset;
logic signed [6:0] shifted;

illa inner (.sinr_q16(sinr_q16), .mcs_index(base_index));
dqn network (.clock, .reset, .start, .observation, .action, .done);

always_comb begin
    case (action)
        3'd0: offset = -6'sd4;
        3'd1: offset = -6'sd3;
        3'd2: offset = -6'sd2;
        3'd3: offset = -6'sd1;
        3'd4: offset = 6'sd0;
        3'd5: offset = 6'sd1;
        default: offset = 6'sd2;
    endcase
    shifted = 7'(base_index) + 7'(offset);
    
    if (shifted < 7'sd3)
        mcs_index = 5'd3;
    else if (shifted > 7'sd28)
        mcs_index = 5'd28;
    else
        mcs_index = 5'(shifted);
end

endmodule
