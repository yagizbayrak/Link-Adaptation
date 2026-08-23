// Checks the learned policy end to end, from a report and an observation to the modulation and coding scheme the trained agent picked.
module tb_dqn_la;

logic clock;
logic reset;
logic start;
logic signed [31:0] sinr_q16;
logic signed [831:0] observation;
logic [4:0] mcs_index;
logic done;

integer handle;
integer entries;
integer failures;
integer status;
integer expected;
integer word;
logic [31:0] value;

dqn_la device (.clock, .reset, .start, .sinr_q16, .observation, .mcs_index, .done);

initial begin
    clock = 1'b0;
    forever #5 clock = ~clock;
end

initial begin
    entries = 0;
    failures = 0;
    reset = 1'b1;
    start = 1'b0;
    sinr_q16 = 32'sd0;
    observation = 832'd0;

    @(posedge clock);
    @(posedge clock);
    reset = 1'b0;

    handle = $fopen("tb/vectors_dqn_la.txt", "r");

    if (handle == 0) begin
        $display("FAIL cannot open tb/vectors_dqn_la.txt");
        $finish;
    end

    while (!$feof(handle)) begin
        status = 0;

        for (word = 0; word < 26; word++) begin
            status = status + $fscanf(handle, "%h", value);
            observation[word*32 +: 32] = value;
        end

        status = status + $fscanf(handle, "%h", value);
        sinr_q16 = value;
        status = status + $fscanf(handle, "%d\n", expected);

        if (status == 28) begin
            @(posedge clock);
            start = 1'b1;
            @(posedge clock);
            #1;
            start = 1'b0;

            wait (done == 1'b1);
            entries = entries + 1;

            if (mcs_index !== expected[4:0]) begin
                failures = failures + 1;

                if (failures <= 5)
                    $display("mismatch sample %0d expected %0d got %0d", entries, expected, mcs_index);
            end

            @(posedge clock);
        end
    end

    $fclose(handle);

    if (failures == 0)
        $display("PASS dqn_la %0d vectors", entries);
    else
        $display("FAIL dqn_la %0d of %0d vectors", failures, entries);

    $finish;
end

endmodule
