//! Compact binary encoding for MDBX storage values.
//!
//! Strips trailing zero bytes from integers and uses bitflags for optional
//! fields, achieving significant space reduction compared to RLP for storage.
//!
//! IMPORTANT: Only used for MDBX storage. Keep RLP for:
//! - Network protocol (P2P, RPC)
//! - Consensus (block hashing, trie computation)
//! - Any non-storage use

use ethrex_common::{
    Address, H256, U256,
    types::{
        AccessList, AuthorizationList, AuthorizationTuple, BlockBody, BlockHeader, Log, Receipt,
        Transaction, TxKind, TxType, Withdrawal,
    },
};
use ethrex_rlp::encode::RLPEncode;
use once_cell::sync::OnceCell;

use crate::error::StoreError;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

pub trait CompactEncode {
    fn compact_encode(&self, buf: &mut Vec<u8>);

    fn to_compact_vec(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.compact_encode(&mut buf);
        buf
    }
}

pub trait CompactDecode: Sized {
    fn compact_decode(buf: &[u8]) -> Result<(Self, &[u8]), StoreError>;

    fn from_compact_vec(buf: &[u8]) -> Result<Self, StoreError> {
        let (val, _) = Self::compact_decode(buf)?;
        Ok(val)
    }
}

// ---------------------------------------------------------------------------
// Primitive helpers
// ---------------------------------------------------------------------------

/// Encode a u64 as: [len_byte, significant_bytes...]
/// len_byte = 0 means value is zero (0 bytes follow).
pub fn encode_u64(val: u64, buf: &mut Vec<u8>) {
    if val == 0 {
        buf.push(0);
        return;
    }
    let be = val.to_be_bytes();
    let leading_zeros = be.iter().take_while(|&&b| b == 0).count();
    let sig_bytes = 8 - leading_zeros;
    buf.push(sig_bytes as u8);
    buf.extend_from_slice(&be[leading_zeros..]);
}

pub fn decode_u64(buf: &[u8]) -> Result<(u64, &[u8]), StoreError> {
    let (&len, rest) = buf.split_first().ok_or(StoreError::DecodeError)?;
    let len = len as usize;
    if len > 8 {
        return Err(StoreError::DecodeError);
    }
    if rest.len() < len {
        return Err(StoreError::DecodeError);
    }
    let (bytes, rest) = rest.split_at(len);
    let mut arr = [0u8; 8];
    arr[8 - len..].copy_from_slice(bytes);
    Ok((u64::from_be_bytes(arr), rest))
}

/// Encode a u32 as: [len_byte, significant_bytes...]
pub fn encode_u32(val: u32, buf: &mut Vec<u8>) {
    if val == 0 {
        buf.push(0);
        return;
    }
    let be = val.to_be_bytes();
    let leading_zeros = be.iter().take_while(|&&b| b == 0).count();
    let sig_bytes = 4 - leading_zeros;
    buf.push(sig_bytes as u8);
    buf.extend_from_slice(&be[leading_zeros..]);
}

pub fn decode_u32(buf: &[u8]) -> Result<(u32, &[u8]), StoreError> {
    let (&len, rest) = buf.split_first().ok_or(StoreError::DecodeError)?;
    let len = len as usize;
    if len > 4 {
        return Err(StoreError::DecodeError);
    }
    if rest.len() < len {
        return Err(StoreError::DecodeError);
    }
    let (bytes, rest) = rest.split_at(len);
    let mut arr = [0u8; 4];
    arr[4 - len..].copy_from_slice(bytes);
    Ok((u32::from_be_bytes(arr), rest))
}

/// Encode a byte slice with a compact u32 length prefix.
pub fn encode_bytes(data: &[u8], buf: &mut Vec<u8>) {
    encode_u32(data.len() as u32, buf);
    buf.extend_from_slice(data);
}

pub fn decode_bytes<'a>(buf: &'a [u8]) -> Result<(&'a [u8], &'a [u8]), StoreError> {
    let (len, rest) = decode_u32(buf)?;
    let len = len as usize;
    if rest.len() < len {
        return Err(StoreError::DecodeError);
    }
    Ok(rest.split_at(len))
}

/// Encode a U256 as: [len_byte, significant_bytes...]
/// Same pattern as encode_u64 but up to 32 bytes.
pub fn encode_u256(val: &U256, buf: &mut Vec<u8>) {
    if val.is_zero() {
        buf.push(0);
        return;
    }
    let be = val.to_big_endian();
    let leading_zeros = be.iter().take_while(|&&b| b == 0).count();
    let sig_bytes = 32 - leading_zeros;
    buf.push(sig_bytes as u8);
    buf.extend_from_slice(&be[leading_zeros..]);
}

pub fn decode_u256(buf: &[u8]) -> Result<(U256, &[u8]), StoreError> {
    let (&len, rest) = buf.split_first().ok_or(StoreError::DecodeError)?;
    let len = len as usize;
    if len > 32 {
        return Err(StoreError::DecodeError);
    }
    if rest.len() < len {
        return Err(StoreError::DecodeError);
    }
    let (bytes, rest) = rest.split_at(len);
    let mut arr = [0u8; 32];
    arr[32 - len..].copy_from_slice(bytes);
    Ok((U256::from_big_endian(&arr), rest))
}

// ---------------------------------------------------------------------------
// Receipt
// ---------------------------------------------------------------------------

/// Receipt compact encoding layout:
///   [tx_type: 1 byte]
///   [flags: 1 byte] bit 0 = succeeded
///   [cumulative_gas_used: compact u64]
///   [logs_count: compact u32]
///   for each log:
///     [address: 20 bytes raw]
///     [topics_count: compact u32]
///     for each topic:
///       [topic: 32 bytes raw]
///     [data: compact bytes]
impl CompactEncode for Receipt {
    fn compact_encode(&self, buf: &mut Vec<u8>) {
        // tx_type byte
        buf.push(self.tx_type as u8);
        // flags: bit 0 = succeeded
        buf.push(self.succeeded as u8);
        // cumulative gas
        encode_u64(self.cumulative_gas_used, buf);
        // logs
        encode_u32(self.logs.len() as u32, buf);
        for log in &self.logs {
            // address: fixed 20 bytes
            buf.extend_from_slice(log.address.as_bytes());
            // topics
            encode_u32(log.topics.len() as u32, buf);
            for topic in &log.topics {
                buf.extend_from_slice(topic.as_bytes());
            }
            // data
            encode_bytes(log.data.as_ref(), buf);
        }
    }
}

impl CompactDecode for Receipt {
    fn compact_decode(buf: &[u8]) -> Result<(Self, &[u8]), StoreError> {
        let (&tx_type_byte, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;
        let tx_type = TxType::from_u8(tx_type_byte).ok_or(StoreError::DecodeError)?;

        let (&flags, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;
        let succeeded = (flags & 1) != 0;

        let (cumulative_gas_used, buf) = decode_u64(buf)?;

        let (log_count, mut buf) = decode_u32(buf)?;
        let mut logs = Vec::with_capacity(log_count as usize);
        for _ in 0..log_count {
            // address
            if buf.len() < 20 {
                return Err(StoreError::DecodeError);
            }
            let (addr_bytes, rest) = buf.split_at(20);
            let address = ethrex_common::Address::from_slice(addr_bytes);
            buf = rest;

            // topics
            let (topic_count, rest) = decode_u32(buf)?;
            buf = rest;
            let mut topics = Vec::with_capacity(topic_count as usize);
            for _ in 0..topic_count {
                if buf.len() < 32 {
                    return Err(StoreError::DecodeError);
                }
                let (topic_bytes, rest) = buf.split_at(32);
                topics.push(H256::from_slice(topic_bytes));
                buf = rest;
            }

            // data
            let (data_bytes, rest) = decode_bytes(buf)?;
            let data = bytes::Bytes::copy_from_slice(data_bytes);
            buf = rest;

            logs.push(Log {
                address,
                topics,
                data,
            });
        }

        Ok((
            Receipt {
                tx_type,
                succeeded,
                cumulative_gas_used,
                logs,
            },
            buf,
        ))
    }
}

// ---------------------------------------------------------------------------
// Transaction location: (BlockNumber, BlockHash, Index)
// ---------------------------------------------------------------------------

/// Encodes (block_number: u64, block_hash: H256, index: u64) compactly:
///   [compact u64: block_number]
///   [32 bytes raw: block_hash]
///   [compact u64: index]
pub fn encode_tx_location(block_number: u64, block_hash: H256, index: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + 8 + 32 + 1 + 8);
    encode_u64(block_number, &mut buf);
    buf.extend_from_slice(block_hash.as_bytes());
    encode_u64(index, &mut buf);
    buf
}

pub fn decode_tx_location(buf: &[u8]) -> Result<(u64, H256, u64), StoreError> {
    let (block_number, buf) = decode_u64(buf)?;
    if buf.len() < 32 {
        return Err(StoreError::DecodeError);
    }
    let (hash_bytes, buf) = buf.split_at(32);
    let block_hash = H256::from_slice(hash_bytes);
    let (index, _) = decode_u64(buf)?;
    Ok((block_number, block_hash, index))
}

// ---------------------------------------------------------------------------
// BlockHeader
// ---------------------------------------------------------------------------

/// BlockHeader compact encoding layout:
///   [parent_hash: 32 bytes raw]
///   [ommers_hash: 32 bytes raw]
///   [coinbase: 20 bytes raw]
///   [state_root: 32 bytes raw]
///   [transactions_root: 32 bytes raw]
///   [receipts_root: 32 bytes raw]
///   [logs_bloom: 256 bytes raw]
///   [difficulty: compact U256]
///   [number: compact u64]
///   [gas_limit: compact u64]
///   [gas_used: compact u64]
///   [timestamp: compact u64]
///   [extra_data: compact bytes (u32 len prefix + raw)]
///   [prev_randao: 32 bytes raw]
///   [nonce: compact u64]
///   [flags: 1 byte] bits 0-7 for the 8 optional fields:
///     bit 0 = base_fee_per_gas present
///     bit 1 = withdrawals_root present
///     bit 2 = blob_gas_used present
///     bit 3 = excess_blob_gas present
///     bit 4 = parent_beacon_block_root present
///     bit 5 = requests_hash present
///     bit 6 = block_access_list_hash present
///     bit 7 = slot_number present
///   [base_fee_per_gas: compact u64] (if bit 0)
///   [withdrawals_root: 32 bytes raw] (if bit 1)
///   [blob_gas_used: compact u64] (if bit 2)
///   [excess_blob_gas: compact u64] (if bit 3)
///   [parent_beacon_block_root: 32 bytes raw] (if bit 4)
///   [requests_hash: 32 bytes raw] (if bit 5)
///   [block_access_list_hash: 32 bytes raw] (if bit 6)
///   [slot_number: compact u64] (if bit 7)
///
/// The `hash: OnceCell<BlockHash>` field is NOT stored (recomputed on demand).
impl CompactEncode for BlockHeader {
    fn compact_encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(self.parent_hash.as_bytes());
        buf.extend_from_slice(self.ommers_hash.as_bytes());
        buf.extend_from_slice(self.coinbase.as_bytes());
        buf.extend_from_slice(self.state_root.as_bytes());
        buf.extend_from_slice(self.transactions_root.as_bytes());
        buf.extend_from_slice(self.receipts_root.as_bytes());
        buf.extend_from_slice(self.logs_bloom.as_bytes());

        encode_u256(&self.difficulty, buf);
        encode_u64(self.number, buf);
        encode_u64(self.gas_limit, buf);
        encode_u64(self.gas_used, buf);
        encode_u64(self.timestamp, buf);

        encode_bytes(self.extra_data.as_ref(), buf);

        buf.extend_from_slice(self.prev_randao.as_bytes());
        encode_u64(self.nonce, buf);

        let mut flags: u8 = 0;
        if self.base_fee_per_gas.is_some()          { flags |= 1 << 0; }
        if self.withdrawals_root.is_some()           { flags |= 1 << 1; }
        if self.blob_gas_used.is_some()              { flags |= 1 << 2; }
        if self.excess_blob_gas.is_some()            { flags |= 1 << 3; }
        if self.parent_beacon_block_root.is_some()   { flags |= 1 << 4; }
        if self.requests_hash.is_some()              { flags |= 1 << 5; }
        if self.block_access_list_hash.is_some()     { flags |= 1 << 6; }
        if self.slot_number.is_some()                { flags |= 1 << 7; }
        buf.push(flags);

        if let Some(v) = self.base_fee_per_gas           { encode_u64(v, buf); }
        if let Some(h) = self.withdrawals_root            { buf.extend_from_slice(h.as_bytes()); }
        if let Some(v) = self.blob_gas_used               { encode_u64(v, buf); }
        if let Some(v) = self.excess_blob_gas             { encode_u64(v, buf); }
        if let Some(h) = self.parent_beacon_block_root    { buf.extend_from_slice(h.as_bytes()); }
        if let Some(h) = self.requests_hash               { buf.extend_from_slice(h.as_bytes()); }
        if let Some(h) = self.block_access_list_hash      { buf.extend_from_slice(h.as_bytes()); }
        if let Some(v) = self.slot_number                 { encode_u64(v, buf); }
    }
}

impl CompactDecode for BlockHeader {
    fn compact_decode(buf: &[u8]) -> Result<(Self, &[u8]), StoreError> {
        if buf.len() < 32 { return Err(StoreError::DecodeError); }
        let (b, buf) = buf.split_at(32);
        let parent_hash = H256::from_slice(b);

        if buf.len() < 32 { return Err(StoreError::DecodeError); }
        let (b, buf) = buf.split_at(32);
        let ommers_hash = H256::from_slice(b);

        if buf.len() < 20 { return Err(StoreError::DecodeError); }
        let (b, buf) = buf.split_at(20);
        let coinbase = Address::from_slice(b);

        if buf.len() < 32 { return Err(StoreError::DecodeError); }
        let (b, buf) = buf.split_at(32);
        let state_root = H256::from_slice(b);

        if buf.len() < 32 { return Err(StoreError::DecodeError); }
        let (b, buf) = buf.split_at(32);
        let transactions_root = H256::from_slice(b);

        if buf.len() < 32 { return Err(StoreError::DecodeError); }
        let (b, buf) = buf.split_at(32);
        let receipts_root = H256::from_slice(b);

        if buf.len() < 256 { return Err(StoreError::DecodeError); }
        let (b, buf) = buf.split_at(256);
        let logs_bloom = ethrex_common::Bloom::from_slice(b);

        let (difficulty, buf) = decode_u256(buf)?;
        let (number, buf) = decode_u64(buf)?;
        let (gas_limit, buf) = decode_u64(buf)?;
        let (gas_used, buf) = decode_u64(buf)?;
        let (timestamp, buf) = decode_u64(buf)?;

        let (extra_data_bytes, buf) = decode_bytes(buf)?;
        let extra_data = bytes::Bytes::copy_from_slice(extra_data_bytes);

        if buf.len() < 32 { return Err(StoreError::DecodeError); }
        let (b, buf) = buf.split_at(32);
        let prev_randao = H256::from_slice(b);

        let (nonce, buf) = decode_u64(buf)?;

        let (&flags, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;

        let (base_fee_per_gas, buf) = if flags & (1 << 0) != 0 {
            let (v, b) = decode_u64(buf)?; (Some(v), b)
        } else { (None, buf) };

        let (withdrawals_root, buf) = if flags & (1 << 1) != 0 {
            if buf.len() < 32 { return Err(StoreError::DecodeError); }
            let (b, rest) = buf.split_at(32);
            (Some(H256::from_slice(b)), rest)
        } else { (None, buf) };

        let (blob_gas_used, buf) = if flags & (1 << 2) != 0 {
            let (v, b) = decode_u64(buf)?; (Some(v), b)
        } else { (None, buf) };

        let (excess_blob_gas, buf) = if flags & (1 << 3) != 0 {
            let (v, b) = decode_u64(buf)?; (Some(v), b)
        } else { (None, buf) };

        let (parent_beacon_block_root, buf) = if flags & (1 << 4) != 0 {
            if buf.len() < 32 { return Err(StoreError::DecodeError); }
            let (b, rest) = buf.split_at(32);
            (Some(H256::from_slice(b)), rest)
        } else { (None, buf) };

        let (requests_hash, buf) = if flags & (1 << 5) != 0 {
            if buf.len() < 32 { return Err(StoreError::DecodeError); }
            let (b, rest) = buf.split_at(32);
            (Some(H256::from_slice(b)), rest)
        } else { (None, buf) };

        let (block_access_list_hash, buf) = if flags & (1 << 6) != 0 {
            if buf.len() < 32 { return Err(StoreError::DecodeError); }
            let (b, rest) = buf.split_at(32);
            (Some(H256::from_slice(b)), rest)
        } else { (None, buf) };

        let (slot_number, buf) = if flags & (1 << 7) != 0 {
            let (v, b) = decode_u64(buf)?; (Some(v), b)
        } else { (None, buf) };

        Ok((
            BlockHeader {
                hash: OnceCell::new(),
                parent_hash,
                ommers_hash,
                coinbase,
                state_root,
                transactions_root,
                receipts_root,
                logs_bloom,
                difficulty,
                number,
                gas_limit,
                gas_used,
                timestamp,
                extra_data,
                prev_randao,
                nonce,
                base_fee_per_gas,
                withdrawals_root,
                blob_gas_used,
                excess_blob_gas,
                parent_beacon_block_root,
                requests_hash,
                block_access_list_hash,
                slot_number,
            },
            buf,
        ))
    }
}

// ---------------------------------------------------------------------------
// TxKind
// ---------------------------------------------------------------------------

fn encode_txkind(kind: &TxKind, buf: &mut Vec<u8>) {
    match kind {
        TxKind::Create => buf.push(0),
        TxKind::Call(addr) => {
            buf.push(1);
            buf.extend_from_slice(addr.as_bytes());
        }
    }
}

fn decode_txkind(buf: &[u8]) -> Result<(TxKind, &[u8]), StoreError> {
    let (&flag, rest) = buf.split_first().ok_or(StoreError::DecodeError)?;
    match flag {
        0 => Ok((TxKind::Create, rest)),
        1 => {
            if rest.len() < 20 {
                return Err(StoreError::DecodeError);
            }
            let (addr_bytes, rest) = rest.split_at(20);
            Ok((TxKind::Call(Address::from_slice(addr_bytes)), rest))
        }
        _ => Err(StoreError::DecodeError),
    }
}

// ---------------------------------------------------------------------------
// AccessList
// ---------------------------------------------------------------------------

fn encode_access_list(list: &AccessList, buf: &mut Vec<u8>) {
    encode_u32(list.len() as u32, buf);
    for (addr, keys) in list {
        buf.extend_from_slice(addr.as_bytes());
        encode_u32(keys.len() as u32, buf);
        for key in keys {
            buf.extend_from_slice(key.as_bytes());
        }
    }
}

fn decode_access_list(buf: &[u8]) -> Result<(AccessList, &[u8]), StoreError> {
    let (count, mut buf) = decode_u32(buf)?;
    let mut list = Vec::with_capacity(count as usize);
    for _ in 0..count {
        if buf.len() < 20 {
            return Err(StoreError::DecodeError);
        }
        let (addr_bytes, rest) = buf.split_at(20);
        let addr = Address::from_slice(addr_bytes);
        buf = rest;

        let (key_count, rest) = decode_u32(buf)?;
        buf = rest;
        let mut keys = Vec::with_capacity(key_count as usize);
        for _ in 0..key_count {
            if buf.len() < 32 {
                return Err(StoreError::DecodeError);
            }
            let (key_bytes, rest) = buf.split_at(32);
            keys.push(H256::from_slice(key_bytes));
            buf = rest;
        }
        list.push((addr, keys));
    }
    Ok((list, buf))
}

// ---------------------------------------------------------------------------
// AuthorizationList
// ---------------------------------------------------------------------------

fn encode_authorization_list(list: &AuthorizationList, buf: &mut Vec<u8>) {
    encode_u32(list.len() as u32, buf);
    for item in list {
        encode_u256(&item.chain_id, buf);
        buf.extend_from_slice(item.address.as_bytes());
        encode_u64(item.nonce, buf);
        encode_u256(&item.y_parity, buf);
        encode_u256(&item.r_signature, buf);
        encode_u256(&item.s_signature, buf);
    }
}

fn decode_authorization_list(buf: &[u8]) -> Result<(AuthorizationList, &[u8]), StoreError> {
    let (count, mut buf) = decode_u32(buf)?;
    let mut list = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let (chain_id, rest) = decode_u256(buf)?;
        buf = rest;
        if buf.len() < 20 {
            return Err(StoreError::DecodeError);
        }
        let (addr_bytes, rest) = buf.split_at(20);
        let address = Address::from_slice(addr_bytes);
        buf = rest;
        let (nonce, rest) = decode_u64(buf)?;
        buf = rest;
        let (y_parity, rest) = decode_u256(buf)?;
        buf = rest;
        let (r_signature, rest) = decode_u256(buf)?;
        buf = rest;
        let (s_signature, rest) = decode_u256(buf)?;
        buf = rest;
        list.push(AuthorizationTuple {
            chain_id,
            address,
            nonce,
            y_parity,
            r_signature,
            s_signature,
        });
    }
    Ok((list, buf))
}

// ---------------------------------------------------------------------------
// Transaction
// ---------------------------------------------------------------------------

/// Transaction compact encoding layout:
///   [variant_byte: 1 byte]  (uses TxType discriminant values)
///   [variant fields...]
///
/// OnceCell fields (inner_hash, sender_cache) are NOT encoded — recomputed on read.
impl CompactEncode for Transaction {
    fn compact_encode(&self, buf: &mut Vec<u8>) {
        match self {
            Transaction::LegacyTransaction(tx) => {
                buf.push(TxType::Legacy as u8);
                encode_u64(tx.nonce, buf);
                encode_u256(&tx.gas_price, buf);
                encode_u64(tx.gas, buf);
                encode_txkind(&tx.to, buf);
                encode_u256(&tx.value, buf);
                encode_bytes(&tx.data, buf);
                encode_u256(&tx.v, buf);
                encode_u256(&tx.r, buf);
                encode_u256(&tx.s, buf);
            }
            Transaction::EIP2930Transaction(tx) => {
                buf.push(TxType::EIP2930 as u8);
                encode_u64(tx.chain_id, buf);
                encode_u64(tx.nonce, buf);
                encode_u256(&tx.gas_price, buf);
                encode_u64(tx.gas_limit, buf);
                encode_txkind(&tx.to, buf);
                encode_u256(&tx.value, buf);
                encode_bytes(&tx.data, buf);
                encode_access_list(&tx.access_list, buf);
                buf.push(tx.signature_y_parity as u8);
                encode_u256(&tx.signature_r, buf);
                encode_u256(&tx.signature_s, buf);
            }
            Transaction::EIP1559Transaction(tx) => {
                buf.push(TxType::EIP1559 as u8);
                encode_u64(tx.chain_id, buf);
                encode_u64(tx.nonce, buf);
                encode_u64(tx.max_priority_fee_per_gas, buf);
                encode_u64(tx.max_fee_per_gas, buf);
                encode_u64(tx.gas_limit, buf);
                encode_txkind(&tx.to, buf);
                encode_u256(&tx.value, buf);
                encode_bytes(&tx.data, buf);
                encode_access_list(&tx.access_list, buf);
                buf.push(tx.signature_y_parity as u8);
                encode_u256(&tx.signature_r, buf);
                encode_u256(&tx.signature_s, buf);
            }
            Transaction::EIP4844Transaction(tx) => {
                buf.push(TxType::EIP4844 as u8);
                encode_u64(tx.chain_id, buf);
                encode_u64(tx.nonce, buf);
                encode_u64(tx.max_priority_fee_per_gas, buf);
                encode_u64(tx.max_fee_per_gas, buf);
                encode_u64(tx.gas, buf);
                buf.extend_from_slice(tx.to.as_bytes());
                encode_u256(&tx.value, buf);
                encode_bytes(&tx.data, buf);
                encode_access_list(&tx.access_list, buf);
                encode_u256(&tx.max_fee_per_blob_gas, buf);
                encode_u32(tx.blob_versioned_hashes.len() as u32, buf);
                for h in &tx.blob_versioned_hashes {
                    buf.extend_from_slice(h.as_bytes());
                }
                buf.push(tx.signature_y_parity as u8);
                encode_u256(&tx.signature_r, buf);
                encode_u256(&tx.signature_s, buf);
            }
            Transaction::EIP7702Transaction(tx) => {
                buf.push(TxType::EIP7702 as u8);
                encode_u64(tx.chain_id, buf);
                encode_u64(tx.nonce, buf);
                encode_u64(tx.max_priority_fee_per_gas, buf);
                encode_u64(tx.max_fee_per_gas, buf);
                encode_u64(tx.gas_limit, buf);
                buf.extend_from_slice(tx.to.as_bytes());
                encode_u256(&tx.value, buf);
                encode_bytes(&tx.data, buf);
                encode_access_list(&tx.access_list, buf);
                encode_authorization_list(&tx.authorization_list, buf);
                buf.push(tx.signature_y_parity as u8);
                encode_u256(&tx.signature_r, buf);
                encode_u256(&tx.signature_s, buf);
            }
            Transaction::PrivilegedL2Transaction(tx) => {
                buf.push(TxType::Privileged as u8);
                encode_u64(tx.chain_id, buf);
                encode_u64(tx.nonce, buf);
                encode_u64(tx.max_priority_fee_per_gas, buf);
                encode_u64(tx.max_fee_per_gas, buf);
                encode_u64(tx.gas_limit, buf);
                encode_txkind(&tx.to, buf);
                encode_u256(&tx.value, buf);
                encode_bytes(&tx.data, buf);
                encode_access_list(&tx.access_list, buf);
                buf.extend_from_slice(tx.from.as_bytes());
            }
            Transaction::FeeTokenTransaction(tx) => {
                buf.push(TxType::FeeToken as u8);
                encode_u64(tx.chain_id, buf);
                encode_u64(tx.nonce, buf);
                encode_u64(tx.max_priority_fee_per_gas, buf);
                encode_u64(tx.max_fee_per_gas, buf);
                encode_u64(tx.gas_limit, buf);
                encode_txkind(&tx.to, buf);
                encode_u256(&tx.value, buf);
                encode_bytes(&tx.data, buf);
                encode_access_list(&tx.access_list, buf);
                buf.extend_from_slice(tx.fee_token.as_bytes());
                buf.push(tx.signature_y_parity as u8);
                encode_u256(&tx.signature_r, buf);
                encode_u256(&tx.signature_s, buf);
            }
        }
    }
}

impl CompactDecode for Transaction {
    fn compact_decode(buf: &[u8]) -> Result<(Self, &[u8]), StoreError> {
        use ethrex_common::types::{
            EIP1559Transaction, EIP2930Transaction, EIP4844Transaction, EIP7702Transaction,
            FeeTokenTransaction, LegacyTransaction, PrivilegedL2Transaction,
        };

        let (&variant, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;
        let tx_type = TxType::from_u8(variant).ok_or(StoreError::DecodeError)?;

        match tx_type {
            TxType::Legacy => {
                let (nonce, buf) = decode_u64(buf)?;
                let (gas_price, buf) = decode_u256(buf)?;
                let (gas, buf) = decode_u64(buf)?;
                let (to, buf) = decode_txkind(buf)?;
                let (value, buf) = decode_u256(buf)?;
                let (data_bytes, buf) = decode_bytes(buf)?;
                let data = bytes::Bytes::copy_from_slice(data_bytes);
                let (v, buf) = decode_u256(buf)?;
                let (r, buf) = decode_u256(buf)?;
                let (s, buf) = decode_u256(buf)?;
                Ok((
                    Transaction::LegacyTransaction(LegacyTransaction {
                        nonce,
                        gas_price,
                        gas,
                        to,
                        value,
                        data,
                        v,
                        r,
                        s,
                        inner_hash: OnceCell::new(),
                        sender_cache: OnceCell::new(),
                    }),
                    buf,
                ))
            }
            TxType::EIP2930 => {
                let (chain_id, buf) = decode_u64(buf)?;
                let (nonce, buf) = decode_u64(buf)?;
                let (gas_price, buf) = decode_u256(buf)?;
                let (gas_limit, buf) = decode_u64(buf)?;
                let (to, buf) = decode_txkind(buf)?;
                let (value, buf) = decode_u256(buf)?;
                let (data_bytes, buf) = decode_bytes(buf)?;
                let data = bytes::Bytes::copy_from_slice(data_bytes);
                let (access_list, buf) = decode_access_list(buf)?;
                let (&sig_y, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;
                let (signature_r, buf) = decode_u256(buf)?;
                let (signature_s, buf) = decode_u256(buf)?;
                Ok((
                    Transaction::EIP2930Transaction(EIP2930Transaction {
                        chain_id,
                        nonce,
                        gas_price,
                        gas_limit,
                        to,
                        value,
                        data,
                        access_list,
                        signature_y_parity: sig_y != 0,
                        signature_r,
                        signature_s,
                        inner_hash: OnceCell::new(),
                        sender_cache: OnceCell::new(),
                    }),
                    buf,
                ))
            }
            TxType::EIP1559 => {
                let (chain_id, buf) = decode_u64(buf)?;
                let (nonce, buf) = decode_u64(buf)?;
                let (max_priority_fee_per_gas, buf) = decode_u64(buf)?;
                let (max_fee_per_gas, buf) = decode_u64(buf)?;
                let (gas_limit, buf) = decode_u64(buf)?;
                let (to, buf) = decode_txkind(buf)?;
                let (value, buf) = decode_u256(buf)?;
                let (data_bytes, buf) = decode_bytes(buf)?;
                let data = bytes::Bytes::copy_from_slice(data_bytes);
                let (access_list, buf) = decode_access_list(buf)?;
                let (&sig_y, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;
                let (signature_r, buf) = decode_u256(buf)?;
                let (signature_s, buf) = decode_u256(buf)?;
                Ok((
                    Transaction::EIP1559Transaction(EIP1559Transaction {
                        chain_id,
                        nonce,
                        max_priority_fee_per_gas,
                        max_fee_per_gas,
                        gas_limit,
                        to,
                        value,
                        data,
                        access_list,
                        signature_y_parity: sig_y != 0,
                        signature_r,
                        signature_s,
                        inner_hash: OnceCell::new(),
                        sender_cache: OnceCell::new(),
                    }),
                    buf,
                ))
            }
            TxType::EIP4844 => {
                let (chain_id, buf) = decode_u64(buf)?;
                let (nonce, buf) = decode_u64(buf)?;
                let (max_priority_fee_per_gas, buf) = decode_u64(buf)?;
                let (max_fee_per_gas, buf) = decode_u64(buf)?;
                let (gas, buf) = decode_u64(buf)?;
                if buf.len() < 20 {
                    return Err(StoreError::DecodeError);
                }
                let (addr_bytes, buf) = buf.split_at(20);
                let to = Address::from_slice(addr_bytes);
                let (value, buf) = decode_u256(buf)?;
                let (data_bytes, buf) = decode_bytes(buf)?;
                let data = bytes::Bytes::copy_from_slice(data_bytes);
                let (access_list, buf) = decode_access_list(buf)?;
                let (max_fee_per_blob_gas, buf) = decode_u256(buf)?;
                let (hash_count, mut buf) = decode_u32(buf)?;
                let mut blob_versioned_hashes = Vec::with_capacity(hash_count as usize);
                for _ in 0..hash_count {
                    if buf.len() < 32 {
                        return Err(StoreError::DecodeError);
                    }
                    let (h_bytes, rest) = buf.split_at(32);
                    blob_versioned_hashes.push(H256::from_slice(h_bytes));
                    buf = rest;
                }
                let (&sig_y, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;
                let (signature_r, buf) = decode_u256(buf)?;
                let (signature_s, buf) = decode_u256(buf)?;
                Ok((
                    Transaction::EIP4844Transaction(EIP4844Transaction {
                        chain_id,
                        nonce,
                        max_priority_fee_per_gas,
                        max_fee_per_gas,
                        gas,
                        to,
                        value,
                        data,
                        access_list,
                        max_fee_per_blob_gas,
                        blob_versioned_hashes,
                        signature_y_parity: sig_y != 0,
                        signature_r,
                        signature_s,
                        inner_hash: OnceCell::new(),
                        sender_cache: OnceCell::new(),
                    }),
                    buf,
                ))
            }
            TxType::EIP7702 => {
                let (chain_id, buf) = decode_u64(buf)?;
                let (nonce, buf) = decode_u64(buf)?;
                let (max_priority_fee_per_gas, buf) = decode_u64(buf)?;
                let (max_fee_per_gas, buf) = decode_u64(buf)?;
                let (gas_limit, buf) = decode_u64(buf)?;
                if buf.len() < 20 {
                    return Err(StoreError::DecodeError);
                }
                let (addr_bytes, buf) = buf.split_at(20);
                let to = Address::from_slice(addr_bytes);
                let (value, buf) = decode_u256(buf)?;
                let (data_bytes, buf) = decode_bytes(buf)?;
                let data = bytes::Bytes::copy_from_slice(data_bytes);
                let (access_list, buf) = decode_access_list(buf)?;
                let (authorization_list, buf) = decode_authorization_list(buf)?;
                let (&sig_y, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;
                let (signature_r, buf) = decode_u256(buf)?;
                let (signature_s, buf) = decode_u256(buf)?;
                Ok((
                    Transaction::EIP7702Transaction(EIP7702Transaction {
                        chain_id,
                        nonce,
                        max_priority_fee_per_gas,
                        max_fee_per_gas,
                        gas_limit,
                        to,
                        value,
                        data,
                        access_list,
                        authorization_list,
                        signature_y_parity: sig_y != 0,
                        signature_r,
                        signature_s,
                        inner_hash: OnceCell::new(),
                        sender_cache: OnceCell::new(),
                    }),
                    buf,
                ))
            }
            TxType::Privileged => {
                let (chain_id, buf) = decode_u64(buf)?;
                let (nonce, buf) = decode_u64(buf)?;
                let (max_priority_fee_per_gas, buf) = decode_u64(buf)?;
                let (max_fee_per_gas, buf) = decode_u64(buf)?;
                let (gas_limit, buf) = decode_u64(buf)?;
                let (to, buf) = decode_txkind(buf)?;
                let (value, buf) = decode_u256(buf)?;
                let (data_bytes, buf) = decode_bytes(buf)?;
                let data = bytes::Bytes::copy_from_slice(data_bytes);
                let (access_list, buf) = decode_access_list(buf)?;
                if buf.len() < 20 {
                    return Err(StoreError::DecodeError);
                }
                let (from_bytes, buf) = buf.split_at(20);
                let from = Address::from_slice(from_bytes);
                Ok((
                    Transaction::PrivilegedL2Transaction(PrivilegedL2Transaction {
                        chain_id,
                        nonce,
                        max_priority_fee_per_gas,
                        max_fee_per_gas,
                        gas_limit,
                        to,
                        value,
                        data,
                        access_list,
                        from,
                        inner_hash: OnceCell::new(),
                        sender_cache: OnceCell::new(),
                    }),
                    buf,
                ))
            }
            TxType::FeeToken => {
                let (chain_id, buf) = decode_u64(buf)?;
                let (nonce, buf) = decode_u64(buf)?;
                let (max_priority_fee_per_gas, buf) = decode_u64(buf)?;
                let (max_fee_per_gas, buf) = decode_u64(buf)?;
                let (gas_limit, buf) = decode_u64(buf)?;
                let (to, buf) = decode_txkind(buf)?;
                let (value, buf) = decode_u256(buf)?;
                let (data_bytes, buf) = decode_bytes(buf)?;
                let data = bytes::Bytes::copy_from_slice(data_bytes);
                let (access_list, buf) = decode_access_list(buf)?;
                if buf.len() < 20 {
                    return Err(StoreError::DecodeError);
                }
                let (fee_token_bytes, buf) = buf.split_at(20);
                let fee_token = Address::from_slice(fee_token_bytes);
                let (&sig_y, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;
                let (signature_r, buf) = decode_u256(buf)?;
                let (signature_s, buf) = decode_u256(buf)?;
                Ok((
                    Transaction::FeeTokenTransaction(FeeTokenTransaction {
                        chain_id,
                        nonce,
                        max_priority_fee_per_gas,
                        max_fee_per_gas,
                        gas_limit,
                        to,
                        value,
                        data,
                        access_list,
                        fee_token,
                        signature_y_parity: sig_y != 0,
                        signature_r,
                        signature_s,
                        inner_hash: OnceCell::new(),
                        sender_cache: OnceCell::new(),
                    }),
                    buf,
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Withdrawal
// ---------------------------------------------------------------------------

impl CompactEncode for Withdrawal {
    fn compact_encode(&self, buf: &mut Vec<u8>) {
        encode_u64(self.index, buf);
        encode_u64(self.validator_index, buf);
        buf.extend_from_slice(self.address.as_bytes());
        encode_u64(self.amount, buf);
    }
}

impl CompactDecode for Withdrawal {
    fn compact_decode(buf: &[u8]) -> Result<(Self, &[u8]), StoreError> {
        let (index, buf) = decode_u64(buf)?;
        let (validator_index, buf) = decode_u64(buf)?;
        if buf.len() < 20 {
            return Err(StoreError::DecodeError);
        }
        let (addr_bytes, buf) = buf.split_at(20);
        let address = Address::from_slice(addr_bytes);
        let (amount, buf) = decode_u64(buf)?;
        Ok((
            Withdrawal {
                index,
                validator_index,
                address,
                amount,
            },
            buf,
        ))
    }
}

// ---------------------------------------------------------------------------
// BlockBody
// ---------------------------------------------------------------------------

/// BlockBody compact encoding layout:
///   [tx_count: compact u32]
///   for each tx: [Transaction compact encoding]
///   [ommers_count: compact u32]
///   for each ommer (always 0 post-merge): [RLP bytes as compact bytes]
///   [has_withdrawals: 1 byte] 0=None, 1=Some
///   if Some: [withdrawal_count: compact u32] [Withdrawal...]
impl CompactEncode for BlockBody {
    fn compact_encode(&self, buf: &mut Vec<u8>) {
        encode_u32(self.transactions.len() as u32, buf);
        for tx in &self.transactions {
            tx.compact_encode(buf);
        }

        encode_u32(self.ommers.len() as u32, buf);
        for ommer in &self.ommers {
            let rlp_bytes = ommer.encode_to_vec();
            encode_bytes(&rlp_bytes, buf);
        }

        match &self.withdrawals {
            None => buf.push(0),
            Some(withdrawals) => {
                buf.push(1);
                encode_u32(withdrawals.len() as u32, buf);
                for w in withdrawals {
                    w.compact_encode(buf);
                }
            }
        }
    }
}

impl CompactDecode for BlockBody {
    fn compact_decode(buf: &[u8]) -> Result<(Self, &[u8]), StoreError> {
        use ethrex_rlp::decode::RLPDecode;

        let (tx_count, mut buf) = decode_u32(buf)?;
        let mut transactions = Vec::with_capacity(tx_count as usize);
        for _ in 0..tx_count {
            let (tx, rest) = Transaction::compact_decode(buf)?;
            transactions.push(tx);
            buf = rest;
        }

        let (ommer_count, mut buf) = decode_u32(buf)?;
        let mut ommers = Vec::with_capacity(ommer_count as usize);
        for _ in 0..ommer_count {
            let (rlp_bytes, rest) = decode_bytes(buf)?;
            let header =
                BlockHeader::decode(rlp_bytes).map_err(|_| StoreError::DecodeError)?;
            ommers.push(header);
            buf = rest;
        }

        let (&has_withdrawals, buf) = buf.split_first().ok_or(StoreError::DecodeError)?;
        if has_withdrawals == 0 {
            return Ok((
                BlockBody {
                    transactions,
                    ommers,
                    withdrawals: None,
                },
                buf,
            ));
        }

        let (w_count, mut buf) = decode_u32(buf)?;
        let mut withdrawals = Vec::with_capacity(w_count as usize);
        for _ in 0..w_count {
            let (w, rest) = Withdrawal::compact_decode(buf)?;
            withdrawals.push(w);
            buf = rest;
        }
        Ok((
            BlockBody {
                transactions,
                ommers,
                withdrawals: Some(withdrawals),
            },
            buf,
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ethrex_common::{Address, H256, types::TxType};

    fn test_receipt(tx_type: TxType, log_count: usize, topic_count: usize) -> Receipt {
        let logs = (0..log_count)
            .map(|i| Log {
                address: Address::from_low_u64_be(i as u64 + 1),
                topics: (0..topic_count)
                    .map(|j| H256::from_low_u64_be(j as u64 + 1))
                    .collect(),
                data: bytes::Bytes::from(vec![0xAB; i * 10]),
            })
            .collect();
        Receipt {
            tx_type,
            succeeded: true,
            cumulative_gas_used: 21000 + log_count as u64 * 1000,
            logs,
        }
    }

    #[test]
    fn receipt_roundtrip_no_logs() {
        let r = Receipt {
            tx_type: TxType::Legacy,
            succeeded: true,
            cumulative_gas_used: 21000,
            logs: vec![],
        };
        let encoded = r.to_compact_vec();
        let decoded = Receipt::from_compact_vec(&encoded).unwrap();
        assert_eq!(r, decoded);
    }

    #[test]
    fn receipt_roundtrip_with_logs() {
        let r = test_receipt(TxType::EIP1559, 3, 4);
        let encoded = r.to_compact_vec();
        let decoded = Receipt::from_compact_vec(&encoded).unwrap();
        assert_eq!(r, decoded);
    }

    #[test]
    fn receipt_roundtrip_failed_tx() {
        let r = Receipt {
            tx_type: TxType::EIP4844,
            succeeded: false,
            cumulative_gas_used: 100_000,
            logs: vec![],
        };
        let encoded = r.to_compact_vec();
        let decoded = Receipt::from_compact_vec(&encoded).unwrap();
        assert_eq!(r, decoded);
    }

    #[test]
    fn receipt_roundtrip_eip2930() {
        let r = test_receipt(TxType::EIP2930, 1, 2);
        let encoded = r.to_compact_vec();
        let decoded = Receipt::from_compact_vec(&encoded).unwrap();
        assert_eq!(r, decoded);
    }

    #[test]
    fn receipt_compact_smaller_than_rlp() {
        use ethrex_rlp::encode::RLPEncode;
        let r = test_receipt(TxType::EIP1559, 5, 4);
        let compact = r.to_compact_vec();
        let rlp = r.encode_to_vec();
        assert!(
            compact.len() < rlp.len(),
            "compact ({}) should be smaller than rlp ({})",
            compact.len(),
            rlp.len()
        );
    }

    #[test]
    fn receipt_compact_smaller_no_logs() {
        use ethrex_rlp::encode::RLPEncode;
        let r = Receipt {
            tx_type: TxType::Legacy,
            succeeded: true,
            cumulative_gas_used: 21000,
            logs: vec![],
        };
        // Compact: 1 (type) + 1 (flags) + compact_u64(21000=0x5208, 2 bytes sig) = 1+1+1+2+1+0 = 6
        let compact = r.to_compact_vec();
        let rlp = r.encode_to_vec();
        // RLP has overhead for list, length prefixes etc.
        // Even without logs compact should be at most as large
        assert!(compact.len() <= rlp.len());
    }

    #[test]
    fn u64_roundtrip_zero() {
        let mut buf = Vec::new();
        encode_u64(0, &mut buf);
        let (val, rest) = decode_u64(&buf).unwrap();
        assert_eq!(val, 0);
        assert!(rest.is_empty());
        assert_eq!(buf.len(), 1); // just the length byte
    }

    #[test]
    fn u64_roundtrip_max() {
        let mut buf = Vec::new();
        encode_u64(u64::MAX, &mut buf);
        let (val, rest) = decode_u64(&buf).unwrap();
        assert_eq!(val, u64::MAX);
        assert!(rest.is_empty());
    }

    #[test]
    fn u64_roundtrip_small() {
        let mut buf = Vec::new();
        encode_u64(1, &mut buf);
        assert_eq!(buf.len(), 2); // len=1 + 1 byte
        let (val, _) = decode_u64(&buf).unwrap();
        assert_eq!(val, 1);
    }

    #[test]
    fn tx_location_roundtrip() {
        let bn = 12_000_000u64;
        let bh = H256::from_low_u64_be(0xdeadbeef);
        let idx = 42u64;

        let encoded = encode_tx_location(bn, bh, idx);
        let (bn2, bh2, idx2) = decode_tx_location(&encoded).unwrap();
        assert_eq!(bn, bn2);
        assert_eq!(bh, bh2);
        assert_eq!(idx, idx2);
    }

    #[test]
    fn tx_location_compact_smaller_than_rlp() {
        use ethrex_rlp::encode::RLPEncode;
        let bn = 12_000_000u64;
        let bh = H256::from_low_u64_be(0xdeadbeef);
        let idx = 100u64;

        let compact = encode_tx_location(bn, bh, idx);
        let rlp = (bn, bh, idx).encode_to_vec();
        assert!(
            compact.len() < rlp.len(),
            "compact ({}) should be smaller than rlp ({})",
            compact.len(),
            rlp.len()
        );
    }

    fn base_header() -> BlockHeader {
        BlockHeader {
            hash: once_cell::sync::OnceCell::new(),
            parent_hash: H256::from_low_u64_be(1),
            ommers_hash: H256::from_low_u64_be(2),
            coinbase: Address::from_low_u64_be(3),
            state_root: H256::from_low_u64_be(4),
            transactions_root: H256::from_low_u64_be(5),
            receipts_root: H256::from_low_u64_be(6),
            logs_bloom: ethrex_common::Bloom::zero(),
            difficulty: U256::zero(),
            number: 100,
            gas_limit: 30_000_000,
            gas_used: 21_000,
            timestamp: 1_700_000_000,
            extra_data: bytes::Bytes::new(),
            prev_randao: H256::from_low_u64_be(7),
            nonce: 0,
            base_fee_per_gas: None,
            withdrawals_root: None,
            blob_gas_used: None,
            excess_blob_gas: None,
            parent_beacon_block_root: None,
            requests_hash: None,
            block_access_list_hash: None,
            slot_number: None,
        }
    }

    #[test]
    fn block_header_roundtrip_pre_shanghai() {
        let h = base_header();
        let encoded = h.to_compact_vec();
        let decoded = BlockHeader::from_compact_vec(&encoded).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn block_header_roundtrip_post_shanghai() {
        let mut h = base_header();
        h.base_fee_per_gas = Some(1_000_000_000);
        h.withdrawals_root = Some(H256::from_low_u64_be(0xabcd));
        let encoded = h.to_compact_vec();
        let decoded = BlockHeader::from_compact_vec(&encoded).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn block_header_roundtrip_post_cancun() {
        let mut h = base_header();
        h.base_fee_per_gas = Some(1_000_000_000);
        h.withdrawals_root = Some(H256::from_low_u64_be(0xabcd));
        h.blob_gas_used = Some(131_072);
        h.excess_blob_gas = Some(0);
        h.parent_beacon_block_root = Some(H256::from_low_u64_be(0xbeef));
        let encoded = h.to_compact_vec();
        let decoded = BlockHeader::from_compact_vec(&encoded).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn block_header_roundtrip_post_prague() {
        let mut h = base_header();
        h.base_fee_per_gas = Some(1_000_000_000);
        h.withdrawals_root = Some(H256::from_low_u64_be(0xabcd));
        h.blob_gas_used = Some(131_072);
        h.excess_blob_gas = Some(262_144);
        h.parent_beacon_block_root = Some(H256::from_low_u64_be(0xbeef));
        h.requests_hash = Some(H256::from_low_u64_be(0xdead));
        let encoded = h.to_compact_vec();
        let decoded = BlockHeader::from_compact_vec(&encoded).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn block_header_roundtrip_post_amsterdam() {
        let mut h = base_header();
        h.base_fee_per_gas = Some(1_000_000_000);
        h.withdrawals_root = Some(H256::from_low_u64_be(0xabcd));
        h.blob_gas_used = Some(131_072);
        h.excess_blob_gas = Some(262_144);
        h.parent_beacon_block_root = Some(H256::from_low_u64_be(0xbeef));
        h.requests_hash = Some(H256::from_low_u64_be(0xdead));
        h.block_access_list_hash = Some(H256::from_low_u64_be(0xcafe));
        h.slot_number = Some(12345);
        let encoded = h.to_compact_vec();
        let decoded = BlockHeader::from_compact_vec(&encoded).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn block_header_roundtrip_with_extra_data() {
        let mut h = base_header();
        h.extra_data = bytes::Bytes::from(vec![0x01, 0x02, 0x03, 0xAB, 0xCD]);
        h.difficulty = U256::from(12345678u64);
        let encoded = h.to_compact_vec();
        let decoded = BlockHeader::from_compact_vec(&encoded).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn block_header_compact_smaller_than_rlp() {
        use ethrex_rlp::encode::RLPEncode;
        let mut h = base_header();
        h.base_fee_per_gas = Some(1_000_000_000);
        h.withdrawals_root = Some(H256::from_low_u64_be(0xabcd));
        h.blob_gas_used = Some(131_072);
        h.excess_blob_gas = Some(0);
        h.parent_beacon_block_root = Some(H256::from_low_u64_be(0xbeef));
        let compact = h.to_compact_vec();
        let rlp = h.encode_to_vec();
        assert!(
            compact.len() < rlp.len(),
            "compact ({}) should be smaller than rlp ({})",
            compact.len(),
            rlp.len()
        );
    }

    fn make_legacy_tx() -> Transaction {
        use ethrex_common::types::LegacyTransaction;
        Transaction::LegacyTransaction(LegacyTransaction {
            nonce: 1,
            gas_price: U256::from(20_000_000_000u64),
            gas: 21000,
            to: TxKind::Call(Address::from_low_u64_be(0xdeadbeef)),
            value: U256::from(1_000_000_000u64),
            data: bytes::Bytes::from(vec![0x01, 0x02, 0x03]),
            v: U256::from(27u64),
            r: U256::from(0x1234u64),
            s: U256::from(0x5678u64),
            inner_hash: OnceCell::new(),
            sender_cache: OnceCell::new(),
        })
    }

    fn make_eip1559_tx() -> Transaction {
        use ethrex_common::types::EIP1559Transaction;
        Transaction::EIP1559Transaction(EIP1559Transaction {
            chain_id: 1,
            nonce: 42,
            max_priority_fee_per_gas: 1_000_000_000,
            max_fee_per_gas: 50_000_000_000,
            gas_limit: 100_000,
            to: TxKind::Call(Address::from_low_u64_be(0xcafe)),
            value: U256::from(0u64),
            data: bytes::Bytes::from(vec![0xde, 0xad]),
            access_list: vec![],
            signature_y_parity: false,
            signature_r: U256::from(0x111u64),
            signature_s: U256::from(0x222u64),
            inner_hash: OnceCell::new(),
            sender_cache: OnceCell::new(),
        })
    }

    fn make_eip1559_create_tx() -> Transaction {
        use ethrex_common::types::EIP1559Transaction;
        Transaction::EIP1559Transaction(EIP1559Transaction {
            chain_id: 1,
            nonce: 0,
            max_priority_fee_per_gas: 0,
            max_fee_per_gas: 0,
            gas_limit: 1_000_000,
            to: TxKind::Create,
            value: U256::zero(),
            data: bytes::Bytes::from(vec![0x60, 0x80, 0x60, 0x40]),
            access_list: vec![],
            signature_y_parity: true,
            signature_r: U256::from(0xaau64),
            signature_s: U256::from(0xbbu64),
            inner_hash: OnceCell::new(),
            sender_cache: OnceCell::new(),
        })
    }

    #[test]
    fn tx_legacy_roundtrip() {
        let tx = make_legacy_tx();
        let encoded = tx.to_compact_vec();
        let decoded = Transaction::from_compact_vec(&encoded).unwrap();
        assert_eq!(tx, decoded);
    }

    #[test]
    fn tx_eip1559_roundtrip() {
        let tx = make_eip1559_tx();
        let encoded = tx.to_compact_vec();
        let decoded = Transaction::from_compact_vec(&encoded).unwrap();
        assert_eq!(tx, decoded);
    }

    #[test]
    fn tx_eip1559_create_roundtrip() {
        let tx = make_eip1559_create_tx();
        let encoded = tx.to_compact_vec();
        let decoded = Transaction::from_compact_vec(&encoded).unwrap();
        assert_eq!(tx, decoded);
    }

    #[test]
    fn tx_eip2930_roundtrip() {
        use ethrex_common::types::EIP2930Transaction;
        let tx = Transaction::EIP2930Transaction(EIP2930Transaction {
            chain_id: 1,
            nonce: 5,
            gas_price: U256::from(10_000_000_000u64),
            gas_limit: 50_000,
            to: TxKind::Call(Address::from_low_u64_be(0x1234)),
            value: U256::from(100u64),
            data: bytes::Bytes::new(),
            access_list: vec![(
                Address::from_low_u64_be(0xabc),
                vec![H256::from_low_u64_be(0x1), H256::from_low_u64_be(0x2)],
            )],
            signature_y_parity: true,
            signature_r: U256::from(0x333u64),
            signature_s: U256::from(0x444u64),
            inner_hash: OnceCell::new(),
            sender_cache: OnceCell::new(),
        });
        let encoded = tx.to_compact_vec();
        let decoded = Transaction::from_compact_vec(&encoded).unwrap();
        assert_eq!(tx, decoded);
    }

    #[test]
    fn tx_eip4844_roundtrip() {
        use ethrex_common::types::EIP4844Transaction;
        let tx = Transaction::EIP4844Transaction(EIP4844Transaction {
            chain_id: 1,
            nonce: 7,
            max_priority_fee_per_gas: 1_000_000,
            max_fee_per_gas: 2_000_000,
            gas: 200_000,
            to: Address::from_low_u64_be(0x9999),
            value: U256::zero(),
            data: bytes::Bytes::new(),
            access_list: vec![],
            max_fee_per_blob_gas: U256::from(1_000_000u64),
            blob_versioned_hashes: vec![
                H256::from_low_u64_be(0xb1),
                H256::from_low_u64_be(0xb2),
            ],
            signature_y_parity: false,
            signature_r: U256::from(0x555u64),
            signature_s: U256::from(0x666u64),
            inner_hash: OnceCell::new(),
            sender_cache: OnceCell::new(),
        });
        let encoded = tx.to_compact_vec();
        let decoded = Transaction::from_compact_vec(&encoded).unwrap();
        assert_eq!(tx, decoded);
    }

    #[test]
    fn tx_eip7702_roundtrip() {
        use ethrex_common::types::{AuthorizationTuple, EIP7702Transaction};
        let tx = Transaction::EIP7702Transaction(EIP7702Transaction {
            chain_id: 1,
            nonce: 3,
            max_priority_fee_per_gas: 500_000,
            max_fee_per_gas: 1_500_000,
            gas_limit: 80_000,
            to: Address::from_low_u64_be(0x7777),
            value: U256::zero(),
            data: bytes::Bytes::from(vec![0xaa]),
            access_list: vec![],
            authorization_list: vec![AuthorizationTuple {
                chain_id: U256::from(1u64),
                address: Address::from_low_u64_be(0x8888),
                nonce: 1,
                y_parity: U256::zero(),
                r_signature: U256::from(0x999u64),
                s_signature: U256::from(0xAAAu64),
            }],
            signature_y_parity: true,
            signature_r: U256::from(0xBBBu64),
            signature_s: U256::from(0xCCCu64),
            inner_hash: OnceCell::new(),
            sender_cache: OnceCell::new(),
        });
        let encoded = tx.to_compact_vec();
        let decoded = Transaction::from_compact_vec(&encoded).unwrap();
        assert_eq!(tx, decoded);
    }

    #[test]
    fn block_body_empty_roundtrip() {
        let body = BlockBody {
            transactions: vec![],
            ommers: vec![],
            withdrawals: Some(vec![]),
        };
        let encoded = body.to_compact_vec();
        let decoded = BlockBody::from_compact_vec(&encoded).unwrap();
        assert_eq!(body, decoded);
    }

    #[test]
    fn block_body_no_withdrawals_roundtrip() {
        let body = BlockBody {
            transactions: vec![make_legacy_tx(), make_eip1559_tx()],
            ommers: vec![],
            withdrawals: None,
        };
        let encoded = body.to_compact_vec();
        let decoded = BlockBody::from_compact_vec(&encoded).unwrap();
        assert_eq!(body, decoded);
    }

    #[test]
    fn block_body_with_withdrawals_roundtrip() {
        let body = BlockBody {
            transactions: vec![make_eip1559_tx(), make_legacy_tx(), make_eip1559_create_tx()],
            ommers: vec![],
            withdrawals: Some(vec![
                Withdrawal {
                    index: 0,
                    validator_index: 100,
                    address: Address::from_low_u64_be(0x1111),
                    amount: 32_000_000_000,
                },
                Withdrawal {
                    index: 1,
                    validator_index: 200,
                    address: Address::from_low_u64_be(0x2222),
                    amount: 16_000_000_000,
                },
            ]),
        };
        let encoded = body.to_compact_vec();
        let decoded = BlockBody::from_compact_vec(&encoded).unwrap();
        assert_eq!(body, decoded);
    }

    #[test]
    fn withdrawal_roundtrip() {
        let w = Withdrawal {
            index: 42,
            validator_index: 999,
            address: Address::from_low_u64_be(0xfeed),
            amount: 1_000_000_000,
        };
        let encoded = w.to_compact_vec();
        let decoded = Withdrawal::from_compact_vec(&encoded).unwrap();
        assert_eq!(w, decoded);
    }
}
