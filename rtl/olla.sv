// Outer loop link adaptation: the inner loop applied to the report corrected by an offset that steps on every acknowledgement.
module olla (
    input logic clock,
    input logic reset,
    input logic feedback_valid,
    input logic ack,
    input logic signed [31:0] sinr_q16,
    output logic [4:0] mcs_index
);

localparam signed [31:0] DELTA_UP_Q16 = 32'sd65536;
localparam signed [31:0] DELTA_DOWN_Q16 = 32'sd7282;
localparam signed [31:0] OFFSET_MIN_Q16 = -32'sd1310720;
localparam signed [31:0] OFFSET_MAX_Q16 = 32'sd1310720;

logic signed [31:0] offset_q16;
logic signed [31:0] lowered_q16;
logic signed [31:0] raised_q16;

assign lowered_q16 = offset_q16 - DELTA_DOWN_Q16;
assign raised_q16 = offset_q16 + DELTA_UP_Q16;

illa inner (.sinr_q16(sinr_q16 - offset_q16), .mcs_index(mcs_index));

always_ff @(posedge clock) begin
    if (reset)
        offset_q16 <= 32'sd0;
    else if (feedback_valid && ack)
        offset_q16 <= (lowered_q16 < OFFSET_MIN_Q16) ? OFFSET_MIN_Q16 : lowered_q16;
    else if (feedback_valid)
        offset_q16 <= (raised_q16 > OFFSET_MAX_Q16) ? OFFSET_MAX_Q16 : raised_q16;
end

endmodule
