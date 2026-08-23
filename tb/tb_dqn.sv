// Feeds recorded observations through the trained network and checks the selected offset matches the one PyTorch chose.
module tb_dqn;

logic clock;
logic reset;
logic start;
logic signed [831:0] observation;
logic [2:0] action;
logic done;

integer handle;
integer entries;
integer failures;
integer status;
integer expected;
integer word;
logic [31:0] value;

dqn device (.clock, .reset, .start, .observation, .action, .done);

initial begin
    clock = 1'b0;
    forever #5 clock = ~clock;
end

initial begin
    entries = 0;
    failures = 0;
    reset = 1'b1;
    start = 1'b0;
    observation = 832'd0;

    @(posedge clock);
    @(posedge clock);
    reset = 1'b0;

    handle = $fopen("tb/vectors_dqn.txt", "r");

    if (handle == 0) begin
        $display("FAIL cannot open tb/vectors_dqn.txt");
        $finish;
    end

    while (!$feof(handle)) begin
        status = 0;

        for (word = 0; word < 26; word++) begin
            status = status + $fscanf(handle, "%h", value);
            observation[word*32 +: 32] = value;
        end

        status = status + $fscanf(handle, "%d\n", expected);

        if (status == 27) begin
            @(posedge clock);
            start = 1'b1;
            @(posedge clock);
            #1;
            start = 1'b0;

            wait (done == 1'b1);
            entries = entries + 1;

            if (action !== expected[2:0]) begin
                failures = failures + 1;

                if (failures <= 5)
                    $display("mismatch sample %0d expected %0d got %0d", entries, expected, action);
            end

            @(posedge clock);
        end
    end

    $fclose(handle);

    if (failures == 0)
        $display("PASS dqn %0d vectors", entries);
    else
        $display("FAIL dqn %0d of %0d vectors", failures, entries);

    $finish;
end

endmodule
