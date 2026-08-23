// Inner loop link adaptation: the highest modulation and coding scheme whose threshold the reported signal to interference and noise ratio clears.
module illa (
    input logic signed [31:0] sinr_q16,
    output logic [4:0] mcs_index
);

logic signed [31:0] threshold [0:25];

initial $readmemh("rtl/illa_thresholds.mem", threshold);

always_comb begin
    mcs_index = 5'd3;

    for (int entry = 0; entry < 26; entry++)
        if (sinr_q16 >= threshold[entry])
            mcs_index = 5'(entry + 3);
end

endmodule
