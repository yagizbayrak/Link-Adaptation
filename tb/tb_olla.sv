// Replays a report and acknowledgement sequence through the outer loop and checks the offset tracks Sionna's choice at every step.
module tb_olla;

logic clock;
logic reset;
logic feedback_valid;
logic ack;
logic signed [31:0] sinr_q16;
logic [4:0] mcs_index;

integer handle;
integer entries;
integer failures;
integer status;
integer expected;
integer feedback;

olla device (.clock, .reset, .feedback_valid, .ack, .sinr_q16, .mcs_index);

initial begin
    clock = 1'b0;
    forever #5 clock = ~clock;
end

initial begin
    entries = 0;
    failures = 0;
    reset = 1'b1;
    feedback_valid = 1'b0;
    ack = 1'b0;
    sinr_q16 = 32'sd0;

    @(posedge clock);
    @(posedge clock);
    reset = 1'b0;

    handle = $fopen("tb/vectors_olla.txt", "r");

    if (handle == 0) begin
        $display("FAIL cannot open tb/vectors_olla.txt");
        $finish;
    end

    while (!$feof(handle)) begin
        status = $fscanf(handle, "%h %d %d\n", sinr_q16, feedback, expected);

        if (status == 3) begin
            #1;
            entries = entries + 1;

            if (mcs_index !== expected[4:0]) begin
                failures = failures + 1;

                if (failures <= 5)
                    $display("mismatch step %0d expected %0d got %0d", entries, expected, mcs_index);
            end

            ack = feedback[0];
            feedback_valid = 1'b1;
            @(posedge clock);
            #1;
            feedback_valid = 1'b0;
        end
    end

    $fclose(handle);

    if (failures == 0)
        $display("PASS olla %0d vectors", entries);
    else
        $display("FAIL olla %0d of %0d vectors", failures, entries);

    $finish;
end

endmodule
