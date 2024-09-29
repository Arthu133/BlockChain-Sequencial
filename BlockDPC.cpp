//
// Created by Dave Nash on 20/10/2017.
//
#include <CL/sycl.hpp>
#include <iostream>
#include <sstream>
#include "Block.h"
#include "sha256.h"

Block::Block(uint32_t nIndexIn, const string &sDataIn) : _nIndex(nIndexIn), _sData(sDataIn)
{
    _nNonce = 0;
    _tTime = time(nullptr);

    sHash = _CalculateHash();
}


using namespace sycl;

void Block::MineBlock(uint32_t nDifficulty)
{
    // Define o dispositivo onde a mineração será executada (GPU, porém CPU em caso de erro)
    gpu_selector gpuSelector;
    queue q(gpuSelector);  // Cria a fila de execução para a GPU


    // Armazena o hash final e o nonce encontrado
    uint32_t nonce_results[1] = {0};
    bool nonce_found[1] = {false};  // Flag para determinar quando o nonce é encontrado

    // Buffer para os resultados
    buffer<uint32_t, 1> nonce_buf(nonce_results, range<1>(1));
    buffer<bool, 1> found_buf(nonce_found, range<1>(1));

    // String de zeros para verificar a dificuldade
    std::string target(nDifficulty, '0');

    // Submeter o kernel para mineração em paralelo
    q.submit([&](handler &h) {
        // Acessar os buffers
        auto nonce_acc = nonce_buf.get_access<access::mode::write>(h);
        auto found_acc = found_buf.get_access<access::mode::write>(h);

        // Executar o kernel paralelo
        h.parallel_for(range<1>(256), [=](id<1> i) {
            uint32_t local_nonce = i[0];

            while (!found_acc[0]) {
                // Concatena os valores para o cálculo do hash
                std::stringstream ss;
                ss << _nIndex << sPrevHash << _tTime << _sData << local_nonce;
                std::string hash = sha256(ss.str());

                // Verifica se o hash gerado atende à dificuldade
                if (hash.substr(0, nDifficulty) == target) {
                    nonce_acc[0] = local_nonce;  // Salva o nonce correto
                    found_acc[0] = true;         // Marca que o nonce foi encontrado
                    break;
                }

                local_nonce += 256;  // Incrementa o nonce em múltiplos de 256 para evitar colisões entre threads
            }
        });
    }).wait();

    // Obter o nonce e gerar o hash final
    _nNonce = nonce_results[0];
    sHash = _CalculateHash();

    std::cout << "Block mined: " << sHash << " with nonce: " << _nNonce << std::endl;
}


inline string Block::_CalculateHash() const
{
    stringstream ss;
    ss << _nIndex << sPrevHash << _tTime << _sData << _nNonce;

    return sha256(ss.str());
}
