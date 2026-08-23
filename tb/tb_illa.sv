// Drives the inner loop with swept reported ratios and compares every choice against the value Sionna produced.
module tb_illa;

logic signed [31:0] sinr_q16;
logic [4:0] mcs_index;

integer handle;
integer entries;
integer failures;
integer status;
integer expected;

illa device (.sinr_q16, .mcs_index);

initial begin
    entries = 0;
    failures = 0;

    handle = $fopen("tb/vectors_illa.txt", "r");

    if (handle == 0) begin
        $display("FAIL cannot open tb/vectors_illa.txt");
        $finish;
    end

    while (!$feof(handle)) begin
        status = $fscanf(handle, "%h %d\n", sinr_q16, expected);

        if (status == 2) begin
            #1;
            entries = entries + 1;

            if (mcs_index !== expected[4:0]) begin
                failures = failures + 1;

                if (failures <= 5)
                    $display("mismatch sinr %h expected %0d got %0d", sinr_q16, expected, mcs_index);
            end
        end
    end

    $fclose(handle);

    if (failures == 0)
        $display("PASS illa %0d vectors", entries);
    else
        $display("FAIL illa %0d of %0d vectors", failures, entries);

    $finish;
end

endmodule
