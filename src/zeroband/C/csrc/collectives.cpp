#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <compression.cpp>

constexpr int BUFFER_COUNT = 2;

template <typename T>
void fast_index_add_omp(T* output, const T* lookup_table, const uint8_t* indices, int64_t n) {
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        output[i] += lookup_table[indices[i]];
    }
}

template <typename T>
void fast_index_set_omp(T* output, const T* lookup_table, const uint8_t* indices, int64_t n) {
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        output[i] = lookup_table[indices[i]];
    }
}

inline size_t get_num_threads() {
    return std::max(1u, std::thread::hardware_concurrency());
}

template <typename T>
void fast_index_add_worker(T* output, const T* lookup_table, const uint8_t* indices, int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
        output[i] += lookup_table[indices[i]];
    }
}

template <typename T>
void fast_index_add(T* output, const T* lookup_table, const uint8_t* indices, int64_t n) {
    size_t num_threads = get_num_threads();
    std::vector<std::thread> threads;
    int64_t chunk_size = n / num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
        threads.emplace_back(fast_index_add_worker<T>, output, lookup_table, indices, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename T>
void fast_index_set_worker(T* output, const T* lookup_table, const uint8_t* indices, int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
        output[i] = lookup_table[indices[i]];
    }
}

template <typename T>
void fast_index_set(T* output, const T* lookup_table, const uint8_t* indices, int64_t n) {
    size_t num_threads = get_num_threads();
    std::vector<std::thread> threads;
    int64_t chunk_size = n / num_threads;

    for (size_t i = 0; i < num_threads; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
        threads.emplace_back(fast_index_set_worker<T>, output, lookup_table, indices, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename T>
void ring_allreduce(
    torch::Tensor& tensor,
    c10d::ReduceOp op,
    T* group
) {
    TORCH_CHECK(group != nullptr, "Group must be provided");
    TORCH_CHECK(op == c10d::ReduceOp::SUM || op == c10d::ReduceOp::AVG, "Unsupported reduce operation. Only SUM and AVG are supported.");

    int world_size = group->getSize();
    int rank = group->getRank();

    // Divide the tensor into chunks
    auto flat_tensor = tensor.view({tensor.numel()});
    std::vector<torch::Tensor> chunks = flat_tensor.chunk(world_size * BUFFER_COUNT);

    // Temporary buffers for transferring data
    int num_buffers = BUFFER_COUNT * world_size;
    std::vector<torch::Tensor> recv_buffer;
    std::vector<torch::Tensor> send_buffer;
    std::vector<torch::Tensor> send_lookup_buffer;
    std::vector<torch::Tensor> recv_lookup_buffer;
    std::vector<c10::intrusive_ptr<c10d::Work>> send_lookup_work(BUFFER_COUNT);
    std::vector<c10::intrusive_ptr<c10d::Work>> recv_lookup_work(BUFFER_COUNT);
    std::vector<c10::intrusive_ptr<c10d::Work>> send_work(BUFFER_COUNT);
    std::vector<c10::intrusive_ptr<c10d::Work>> recv_work(BUFFER_COUNT);

    for (int i = 0; i < BUFFER_COUNT; ++i) {
        recv_buffer.push_back(torch::empty_like(chunks[0], torch::kUInt8));
        send_buffer.push_back(torch::Tensor());
        send_lookup_buffer.push_back(torch::Tensor());
        recv_lookup_buffer.push_back(torch::empty({256}, chunks[0].options()));
    }

    // Send and receive ranks
    int send_rank = (rank + 1) % world_size;
    int recv_rank = (rank - 1 + world_size) % world_size;

    // Reduce-scatter loop
    for (int step = 1; step <= world_size * BUFFER_COUNT; ++step) {
        int send_chunk = (rank * BUFFER_COUNT - step + num_buffers) % num_buffers;

        if (send_work[step % BUFFER_COUNT]) {
            send_work[step % BUFFER_COUNT]->wait();
            recv_work[step % BUFFER_COUNT]->wait();
            send_lookup_work[step % BUFFER_COUNT]->wait();
            recv_lookup_work[step % BUFFER_COUNT]->wait();

            auto& chunk = chunks[send_chunk];
            auto& lookup = recv_lookup_buffer[step % BUFFER_COUNT];
            auto& indices = recv_buffer[step % BUFFER_COUNT];

            fast_index_add_omp<float>(
                static_cast<float*>(chunk.data_ptr()),
                static_cast<const float*>(lookup.data_ptr()),
                static_cast<const uint8_t*>(indices.data_ptr()),
                chunk.numel()
            );
        }

        if (step <= (world_size - 1) * BUFFER_COUNT) {
            // Quantize and send
            std::tie(send_buffer[step % BUFFER_COUNT], send_lookup_buffer[step % BUFFER_COUNT]) = uniform_8bit_quantize(chunks[send_chunk], false);

            std::vector<torch::Tensor> send_tensors = {send_lookup_buffer[step % BUFFER_COUNT]};
            send_lookup_work[step % BUFFER_COUNT] = group->send(send_tensors, send_rank, step + 1000);

            std::vector<torch::Tensor> recv_tensors = {recv_lookup_buffer[step % BUFFER_COUNT]};
            recv_lookup_work[step % BUFFER_COUNT] = group->recv(recv_tensors, recv_rank, step + 1000);

            send_tensors = {send_buffer[step % BUFFER_COUNT]};
            send_work[step % BUFFER_COUNT] = group->send(send_tensors, send_rank, step);

            recv_tensors = {recv_buffer[step % BUFFER_COUNT]};
            recv_work[step % BUFFER_COUNT] = group->recv(recv_tensors, recv_rank, step);
        }
    }

    // TODO: Interleave these with the previous loop?
    if (op == c10d::ReduceOp::AVG) {
        for (int i = 0; i < BUFFER_COUNT; ++i) {
            chunks[i + rank * BUFFER_COUNT].div_(world_size);
        }
    }
    
    // Quantize your chunk to be consistent
    for (int step = 1; step <= BUFFER_COUNT; ++step) {
        std::tie(send_buffer[step % BUFFER_COUNT], send_lookup_buffer[step % BUFFER_COUNT]) = uniform_8bit_quantize(chunks[BUFFER_COUNT - step + rank * BUFFER_COUNT], true);
        auto& chunk = chunks[BUFFER_COUNT - step + rank * BUFFER_COUNT];
        auto& lookup = send_lookup_buffer[step % BUFFER_COUNT];
        auto& indices = send_buffer[step % BUFFER_COUNT];

        fast_index_set_omp<float>(
            static_cast<float*>(chunk.data_ptr()),
            static_cast<const float*>(lookup.data_ptr()),
            static_cast<const uint8_t*>(indices.data_ptr()),
            chunk.numel()
        );
    }

    // Reset buffers for the second phase
    recv_buffer.clear();
    //send_buffer.clear();
    //send_lookup_buffer.clear();
    recv_lookup_buffer.clear();
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        recv_buffer.push_back(torch::empty_like(chunks[0], torch::kUInt8));
        //send_buffer.push_back(torch::Tensor());
        //send_lookup_buffer.push_back(torch::Tensor());
        recv_lookup_buffer.push_back(torch::empty({256}, chunks[0].options()));
    }
    std::fill(send_work.begin(), send_work.end(), nullptr);
    std::fill(recv_work.begin(), recv_work.end(), nullptr);
    std::fill(send_lookup_work.begin(), send_lookup_work.end(), nullptr);
    std::fill(recv_lookup_work.begin(), recv_lookup_work.end(), nullptr);

    for (int step = 1; step <= world_size * BUFFER_COUNT; ++step) {
        int send_chunk = (rank * BUFFER_COUNT + BUFFER_COUNT - step + num_buffers) % num_buffers;

        if (send_work[step % BUFFER_COUNT]) {
            send_work[step % BUFFER_COUNT]->wait();
            recv_work[step % BUFFER_COUNT]->wait();
            send_lookup_work[step % BUFFER_COUNT]->wait();
            recv_lookup_work[step % BUFFER_COUNT]->wait();

            auto& chunk = chunks[send_chunk];
            auto& lookup = recv_lookup_buffer[step % BUFFER_COUNT];
            auto& indices = recv_buffer[step % BUFFER_COUNT];

            fast_index_set_omp<float>(
                static_cast<float*>(chunk.data_ptr()),
                static_cast<const float*>(lookup.data_ptr()),
                static_cast<const uint8_t*>(indices.data_ptr()),
                chunk.numel()
            );
        }

        if (step <= (world_size - 1) * BUFFER_COUNT) {
            // Quantize and send
            // todo(jackmin): this copy breaks things
            // todo(jackmin): we can also go even faster by not copying at all
            if (step > BUFFER_COUNT) {
                send_buffer[step % BUFFER_COUNT].copy_(recv_buffer[step % BUFFER_COUNT]);
                send_lookup_buffer[step % BUFFER_COUNT].copy_(recv_lookup_buffer[step % BUFFER_COUNT]);
            }

            std::vector<torch::Tensor> send_tensors = {send_lookup_buffer[step % BUFFER_COUNT]};
            send_lookup_work[step % BUFFER_COUNT] = group->send(send_tensors, send_rank, step + 1000);

            std::vector<torch::Tensor> recv_tensors = {recv_lookup_buffer[step % BUFFER_COUNT]};
            recv_lookup_work[step % BUFFER_COUNT] = group->recv(recv_tensors, recv_rank, step + 1000);

            send_tensors = {send_buffer[step % BUFFER_COUNT]};
            send_work[step % BUFFER_COUNT] = group->send(send_tensors, send_rank, step);

            recv_tensors = {recv_buffer[step % BUFFER_COUNT]};
            recv_work[step % BUFFER_COUNT] = group->recv(recv_tensors, recv_rank, step);
        }
    }
}

PYBIND11_MODULE(collectives, m) {
    m.def(
        "ring_allreduce",
        &ring_allreduce<c10d::ProcessGroup>,
        "Ring allreduce implementation",
        py::arg("tensor"),
        py::arg("op"),
        py::arg("pg")
    );
    m.def(
        "ring_allreduce_gloo",
        &ring_allreduce<c10d::ProcessGroupGloo>,
        "Ring allreduce implementation",
        py::arg("tensor"),
        py::arg("op"),
        py::arg("pg")
    );
}