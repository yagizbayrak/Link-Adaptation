// Trained policy network: a twenty six input, two hidden layer perceptron streamed through a pipelined multiply accumulate against block memory resident weights.
module dqn (
    input logic clock,
    input logic reset,
    input logic start,
    input logic signed [831:0] observation,
    output logic [2:0] action,
    output logic done
);

localparam int INPUTS = 26;
localparam int HIDDEN = 256;
localparam int OUTPUTS = 7;
localparam int DRAIN_CYCLES = 6;
localparam int IDLE = 0;
localparam int ISSUE = 1;
localparam int DRAIN = 2;
localparam int STORE = 3;
localparam int PICK = 4;
localparam int FINISH = 5;

logic signed [31:0] weight1 [0:8191];
logic signed [31:0] weight2 [0:65535];
logic signed [31:0] weight3 [0:2047];
logic signed [31:0] bias1 [0:255];
logic signed [31:0] bias2 [0:255];
logic signed [31:0] bias3 [0:6];

logic signed [31:0] hidden1 [0:255];
logic signed [31:0] hidden2 [0:255];
logic signed [31:0] output3 [0:6];

logic [2:0] state;
logic [1:0] layer;
logic [7:0] neuron;
logic [7:0] tap;
logic [2:0] drain;
logic signed [31:0] best_value;

logic [15:0] rom_address;
logic signed [31:0] weight1_data;
logic signed [31:0] weight2_data;
logic signed [31:0] weight3_data;
logic signed [31:0] weight_data;
logic signed [31:0] operand_stage1;
logic signed [47:0] product_stage1;
logic signed [47:0] product_stage2;
logic signed [47:0] product_stage3;
logic signed [47:0] accumulator;
logic signed [47:0] biased;
logic signed [31:0] activation;
logic [3:0] valid;

logic [8:0] taps_in_layer;
logic [8:0] neurons_in_layer;
logic signed [31:0] operand_source;
logic signed [31:0] bias_source;

initial begin
    $readmemh("rtl/dqn_w1.mem", weight1);
    $readmemh("rtl/dqn_b1.mem", bias1);
    $readmemh("rtl/dqn_w2.mem", weight2);
    $readmemh("rtl/dqn_b2.mem", bias2);
    $readmemh("rtl/dqn_w3.mem", weight3);
    $readmemh("rtl/dqn_b3.mem", bias3);
end

always_comb begin
    taps_in_layer = (layer == 2'd1) ? 9'(INPUTS) : 9'(HIDDEN);
    neurons_in_layer = (layer == 2'd3) ? 9'(OUTPUTS) : 9'(HIDDEN);

    if (layer == 2'd1)
        operand_source = observation[tap*32 +: 32];
    else if (layer == 2'd2)
        operand_source = hidden1[tap];
    else
        operand_source = hidden2[tap];

    if (layer == 2'd1)
        bias_source = bias1[neuron];
    else if (layer == 2'd2)
        bias_source = bias2[neuron];
    else
        bias_source = bias3[neuron[2:0]];

    if (layer == 2'd1)
        weight_data = weight1_data;
    else if (layer == 2'd2)
        weight_data = weight2_data;
    else
        weight_data = weight3_data;

    biased = 48'(signed'(bias_source)) <<< 16;
    activation = 32'(accumulator >>> 16);
end

always_ff @(posedge clock) begin
    weight1_data <= weight1[rom_address[12:0]];
    weight2_data <= weight2[rom_address];
    weight3_data <= weight3[rom_address[10:0]];

    product_stage1 <= 48'(operand_stage1 * weight_data);
    product_stage2 <= product_stage1;
    product_stage3 <= product_stage2;

    if (reset)
        valid <= 4'd0;
    else
        valid <= {valid[2:0], (state == ISSUE)};

    if (reset)
        accumulator <= 48'sd0;
    else if (state == ISSUE && tap == 8'd0)
        accumulator <= biased;
    else if (valid[3])
        accumulator <= accumulator + product_stage3;
end

always_ff @(posedge clock) begin
    if (reset) begin
        state <= IDLE;
        done <= 1'b0;
        action <= 3'd0;
        layer <= 2'd1;
        neuron <= 8'd0;
        tap <= 8'd0;
        drain <= 3'd0;
    end else begin
        case (state)
            IDLE:
                if (start) begin
                    state <= ISSUE;
                    layer <= 2'd1;
                    neuron <= 8'd0;
                    tap <= 8'd0;
                    rom_address <= 16'd0;
                    done <= 1'b0;
                end

            ISSUE: begin
                rom_address <= rom_address + 16'd1;
                operand_stage1 <= operand_source;

                if (tap == taps_in_layer - 9'd1) begin
                    state <= DRAIN;
                    drain <= 3'd0;
                end else begin
                    tap <= tap + 8'd1;
                end
            end

            DRAIN:
                if (drain == DRAIN_CYCLES - 1)
                    state <= STORE;
                else
                    drain <= drain + 3'd1;

            STORE: begin
                if (layer == 2'd1)
                    hidden1[neuron] <= (activation < 0) ? 32'sd0 : activation;
                else if (layer == 2'd2)
                    hidden2[neuron] <= (activation < 0) ? 32'sd0 : activation;
                else
                    output3[neuron[2:0]] <= activation;

                tap <= 8'd0;

                if (neuron == neurons_in_layer - 9'd1) begin
                    neuron <= 8'd0;

                    if (layer == 2'd3) begin
                        state <= PICK;
                    end else begin
                        layer <= layer + 2'd1;
                        rom_address <= 16'd0;
                        state <= ISSUE;
                    end
                end else begin
                    neuron <= neuron + 8'd1;
                    state <= ISSUE;
                end
            end

            PICK: begin
                if (neuron == 8'd0 || output3[neuron[2:0]] > best_value) begin
                    best_value <= output3[neuron[2:0]];
                    action <= neuron[2:0];
                end

                if (neuron == OUTPUTS - 1)
                    state <= FINISH;
                else
                    neuron <= neuron + 8'd1;
            end

            FINISH: begin
                done <= 1'b1;
                state <= IDLE;
            end

            default:
                state <= IDLE;
        endcase
    end
end

endmodule
