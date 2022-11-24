#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <cstdint>
#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pagmo/utils/multi_objective.hpp>

using Factors = std::vector<size_t>;
using PartialKeys = std::vector<size_t>;
using PartialValues = std::vector<size_t>;
using PartialFactors = std::pair<PartialKeys, PartialValues>;
using Vector = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;
/**
 * @brief This class enumerates all possible values for a PartialFactors.
 *
 * This class is a simple enumerator that goes through all possible
 * values of a PartialFactors for the specific input factors. An
 * additional separate factor index can be specified in order to skip
 * that factor, to allow the user to modify that freely.
 *
 * The iteration is *always* done by increasing the lowest id first. So for
 * example in a space (2,3), we iterate in the following order:
 *
 * (0,0)
 * (1,0)
 * (0,1)
 * (1,1)
 * (0,2)
 * (1,2)
 */
class PartialFactorsEnumerator {
    public:
        /**
         * @brief Basic constructor.
         *
         * This constructor initializes the internal PartialFactors
         * with the factors obtained as inputs. In addition it saves
         * the input Factor as the ceiling for the values in the
         * PartialFactors.
         *
         * @param f The factor space for the internal PartialFactors.
         * @param factors The factors to take into consideration.
         */
        PartialFactorsEnumerator(Factors f, PartialKeys factors);

        /**
         * @brief Basic constructor.
         *
         * This constructor initializes the internal PartialFactors with
         * the factors obtained as inputs. This constructor can be used
         * when one wants to iterate over all factors.
         *
         * @param f The factor space for the internal PartialFactors.
         */
        PartialFactorsEnumerator(Factors f);

        /**
         * @brief Skip constructor.
         *
         * This constructor is the same as the basic one, but it
         * additionally remembers that the input factorToSkip will not
         * be enumerated, and will in fact be editable by the client.
         *
         * The factorToSkip must be within the Factors space, or it
         * will not be taken into consideration.
         *
         * @param f The factor space for the internal PartialFactors.
         * @param factors The factors to take into considerations.
         * @param factorToSkip The factor to skip.
         * @param missing Whether factorToSkip is already present in the input PartialKeys or it must be added.
         */
        PartialFactorsEnumerator(Factors f, const PartialKeys & factors, size_t factorToSkip, bool missing = false);

        /**
         * @brief Skip constructor.
         *
         * This constructor is the same as the basic one, but it
         * additionally remembers that the input factorToSkip will not
         * be enumerated, and will in fact be editable by the client.
         *
         * This constructor can be used to enumerate over all factors.
         *
         * The factorToSkip must be within the Factors space, or it
         * will not be taken into consideration.
         *
         * @param f The factor space for the internal PartialFactors.
         * @param factorToSkip The factor to skip.
         */
        PartialFactorsEnumerator(Factors f, size_t factorToSkip);

        /**
         * @brief This function returns the id of the factorToSkip inside the PartialFactorsEnumerator.
         *
         * This function is provided for convenience, since
         * PartialFactorsEnumerator has to compute this id anyway. It
         * represents the id of the factorToSkip inside the vectors
         * contained in the PartialFactors. This is useful so the
         * client can go and edit that particular element directly.
         *
         * @return The id of the factorToSkip inside the PartialFactorsEnumerator.
         */
        size_t getFactorToSkipId() const;

        /**
         * @brief This function advances the PartialFactorsEnumerator to the next possible combination.
         */
        void advance();

        /**
         * @brief This function returns whether this object has terminated advancing and can be dereferenced.
         *
         * @return True if we can still be dereferenced, false otherwise.
         */
        bool isValid() const;

        /**
         * @brief This function resets the enumerator to the valid beginning (a fully zero PartialFactor).
         */
        void reset();

        /**
         * @brief This function returns the number of times that advance() can be called from the initial state.
         *
         * Warning: This operation is *NOT* cheap, as this number needs to be computed.
         */
        size_t size() const;

        /**
         * @brief This operator returns the current iteration in the values of the PartialFactors.
         *
         * This operator can be called only if isValid() is true.
         * Otherwise behavior is undefined.
         *
         * The PartialFactors returned are editable, so that the user
         * can change the factorToSkip values. If other values are
         * edited, and the resulting PartialFactors still has valid
         * values (below the Factors ceiling), the next advance() call
         * will continue from there. If advance() is called with an
         * invalid PartialFactors behavior is undefined.
         *
         * @return The current PartialFactors values.
         */
        PartialFactors& operator*();

        /**
         * @brief This operator returns the current iteration in the values of the PartialFactors.
         *
         * This operator can be called only if isValid() is true.
         * Otherwise behavior is undefined.
         *
         * The PartialFactors returned are editable, so that the user
         * can change the factorToSkip values. If other values are
         * edited, and the resulting PartialFactors still has valid
         * values (below the Factors ceiling), the next advance() call
         * will continue from there. If advance() is called with an
         * invalid PartialFactors behavior is undefined.
         *
         * @return The current PartialFactors values.
         */
        PartialFactors* operator->();

    private:
        Factors F;
        PartialFactors factors_;
        size_t factorToSkipId_;
};

PartialFactorsEnumerator::PartialFactorsEnumerator(Factors f, PartialKeys factors) :
    F(std::move(f)), factorToSkipId_(factors.size())
{
    factors_.first = std::move(factors);
    factors_.second.resize(factors_.first.size());
}

PartialFactorsEnumerator::PartialFactorsEnumerator(Factors f) :
    F(std::move(f)), factorToSkipId_(F.size())
{
    factors_.first.resize(F.size());
    std::iota(std::begin(factors_.first), std::end(factors_.first), 0);
    factors_.second.resize(factors_.first.size());
}

PartialFactorsEnumerator::PartialFactorsEnumerator(Factors f, const PartialKeys & factors, const size_t factorToSkip, bool missing) :
    F(std::move(f))
{
    if (!missing) {
        factors_.first = factors;

        // Find the skip id.
        for (size_t i = 0; i < factors_.first.size(); ++i) {
            if (factorToSkip == factors_.first[i]) {
                factorToSkipId_ = i;
                break;
            }
        }
    } else {
        factors_.first.resize(factors.size() + 1);
        for (size_t i = 0, j = 0; j < factors_.first.size(); ) {
            if (i == j && (i == factors.size() || factors[i] > factorToSkip)) {
                factors_.first[j] = factorToSkip;
                factorToSkipId_ = j;
                ++j;
            } else {
                factors_.first[j] = factors[i];
                ++i, ++j;
            }
        }
    }
    factors_.second.resize(factors_.first.size());
}

PartialFactorsEnumerator::PartialFactorsEnumerator(Factors f, const size_t factorToSkip) :
    PartialFactorsEnumerator(std::move(f))
{
    factorToSkipId_ = factorToSkip;
}

void PartialFactorsEnumerator::advance() {
    // Start from 0 if skip is not zero, from 1 otherwise.
    size_t id = !factorToSkipId_;
    while (id < factors_.second.size()) {
        ++factors_.second[id];
        if (factors_.second[id] == F[factors_.first[id]]) {
            factors_.second[id] = 0;
            if (++id == factorToSkipId_) ++id;
        } else
            return;
    }
    factors_.second.clear();
}

bool PartialFactorsEnumerator::isValid() const {
    return factors_.second.size() > 0;
}

void PartialFactorsEnumerator::reset() {
    if (factors_.second.size() == 0)
        factors_.second.resize(factors_.first.size());
    else
        std::fill(std::begin(factors_.second), std::end(factors_.second), 0);
}

size_t PartialFactorsEnumerator::size() const {
    size_t retval = factors_.first.size() > 0;
    for (size_t i = 0; i < factors_.first.size(); ++i) {
        if (i == factorToSkipId_) continue;
        retval *= F[factors_.first[i]];
    }
    return retval;
}

size_t PartialFactorsEnumerator::getFactorToSkipId() const { return factorToSkipId_; }

PartialFactors& PartialFactorsEnumerator::operator*() { return factors_; }
PartialFactors* PartialFactorsEnumerator::operator->() { return &factors_; }

Vector initializeGrid(const size_t size, const size_t dim, const size_t seed) {
    const size_t totalSize = std::pow(size, dim);
    // std::cout << "Size = " << size << "; Dim = " << dim << "; Total size = " << totalSize << '\n';

    auto F = Factors(dim, size);

    Vector data(totalSize);
    // std::vector<std::vector<double>> coordsToPurge;

    PartialFactorsEnumerator e(F);
    auto distNorm = std::normal_distribution<double>(0.0, size / 10.0);
    auto rnd = std::default_random_engine(seed);

    size_t i = 0;
    while (e.isValid()) {
        double pos = 0.;
        // std::cout << "For coordinate: ";
        std::vector<double> coord; // COORDINATE FOR PAGMO, WHATEVER VECTOR TYPE YOU WANT
        coord.reserve(dim+1);      // RESERVE MEMORY FOR !FAST!
        for (auto x : e->second) {
            // std::cout << x << ", ";
            pos += x; // * 1./(dim-1);
            // coord.push_back(static_cast<double>(x)); // KEEP THE STATIC CAST
        }
        pos += distNorm(rnd);
        pos = size - 1 - pos;

        size_t posI = std::lround(std::max(0.0, std::min(double(size-1), pos)));
        data[i++] = posI;
        // std::cout << " I get " << posI << " (" << pos << ")\n";

        // coord.push_back(static_cast<double>(posI));
        // coordsToPurge.push_back(std::move(coord)); // THE MOVE IS TO DON'T DO REALLOCATIONS

        e.advance();
    }

    // auto ranks = std::get<0>(pagmo::fast_non_dominated_sorting(coordsToPurge));
    // auto nd = ranks[0];
    // for (auto const& v : nd) {
    //     std::cout << v << ' ';
    // }
    // std::cout << '\n';
    // auto k = 0;
    // for (size_t i = 0; i < size; ++i) {
    //     for (size_t j = 0; j < size; ++j)
    //         std::cout << data[k++] << ' ';
    //     std::cout << '\n';
    // }
    return data; // std::make_tuple(data, nd);
}

PYBIND11_MODULE(grid, m) {
    m.def("initializeGrid", &initializeGrid);
}
