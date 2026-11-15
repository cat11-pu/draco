'use strict';

const fs = require('fs');
const path = require('path');

const draco3d = require(path.resolve(
    __dirname, '../../../javascript/npm/draco3d/draco3d'));

function byteToHex(value) {
  return value.toString(16).padStart(2, '0');
}

function valueToHex(module, dataType, value) {
  if (dataType === module.DT_FLOAT32) {
    const out = Buffer.alloc(4);
    out.writeFloatLE(value, 0);
    return out.toString('hex');
  }
  if (dataType === module.DT_FLOAT64) {
    const out = Buffer.alloc(8);
    out.writeDoubleLE(value, 0);
    return out.toString('hex');
  }
  if (dataType === module.DT_INT8 || dataType === module.DT_UINT8 ||
      dataType === module.DT_BOOL) {
    return byteToHex(value & 0xff);
  }
  if (dataType === module.DT_INT16) {
    const out = Buffer.alloc(2);
    out.writeInt16LE(value, 0);
    return out.toString('hex');
  }
  if (dataType === module.DT_UINT16) {
    const out = Buffer.alloc(2);
    out.writeUInt16LE(value, 0);
    return out.toString('hex');
  }
  if (dataType === module.DT_INT32) {
    const out = Buffer.alloc(4);
    out.writeInt32LE(value, 0);
    return out.toString('hex');
  }
  if (dataType === module.DT_UINT32) {
    const out = Buffer.alloc(4);
    out.writeUInt32LE(value, 0);
    return out.toString('hex');
  }
  if (dataType === module.DT_INT64) {
    const out = Buffer.alloc(8);
    out.writeBigInt64LE(BigInt(value), 0);
    return out.toString('hex');
  }
  if (dataType === module.DT_UINT64) {
    const out = Buffer.alloc(8);
    out.writeBigUInt64LE(BigInt(value), 0);
    return out.toString('hex');
  }
  throw new Error(`Unsupported data type ${dataType}`);
}

function readAttributeValues(module, decoder, mesh, attribute) {
  const dataType = attribute.data_type();
  if (dataType === module.DT_FLOAT32) {
    const values = new module.DracoFloat32Array();
    decoder.GetAttributeFloatForAllPoints(mesh, attribute, values);
    return {values, destroy: () => module.destroy(values)};
  }
  if (dataType === module.DT_INT8) {
    const values = new module.DracoInt8Array();
    decoder.GetAttributeInt8ForAllPoints(mesh, attribute, values);
    return {values, destroy: () => module.destroy(values)};
  }
  if (dataType === module.DT_UINT8 || dataType === module.DT_BOOL) {
    const values = new module.DracoUInt8Array();
    decoder.GetAttributeUInt8ForAllPoints(mesh, attribute, values);
    return {values, destroy: () => module.destroy(values)};
  }
  if (dataType === module.DT_INT16) {
    const values = new module.DracoInt16Array();
    decoder.GetAttributeInt16ForAllPoints(mesh, attribute, values);
    return {values, destroy: () => module.destroy(values)};
  }
  if (dataType === module.DT_UINT16) {
    const values = new module.DracoUInt16Array();
    decoder.GetAttributeUInt16ForAllPoints(mesh, attribute, values);
    return {values, destroy: () => module.destroy(values)};
  }
  if (dataType === module.DT_INT32) {
    const values = new module.DracoInt32Array();
    decoder.GetAttributeInt32ForAllPoints(mesh, attribute, values);
    return {values, destroy: () => module.destroy(values)};
  }
  if (dataType === module.DT_UINT32) {
    const values = new module.DracoUInt32Array();
    decoder.GetAttributeUInt32ForAllPoints(mesh, attribute, values);
    return {values, destroy: () => module.destroy(values)};
  }
  throw new Error(`Unsupported data type ${dataType}`);
}

function dumpModel(module, decoder, decoded, seed) {
  const lines = [];
  lines.push(`MODEL ${seed}`);
  lines.push(`faces ${decoded.num_faces()}`);
  lines.push(`points ${decoded.num_points()}`);
  lines.push(`attrs ${decoded.num_attributes()}`);

  const entries = [];
  for (let attId = 0; attId < decoded.num_attributes(); ++attId) {
    const attribute = decoder.GetAttribute(decoded, attId);
    entries.push({
      attribute,
      type: attribute.attribute_type(),
      uniqueId: attribute.unique_id()
    });
  }
  entries.sort((left, right) => {
    if (left.type !== right.type) {
      return left.type - right.type;
    }
    return left.uniqueId - right.uniqueId;
  });

  for (const entry of entries) {
    const attribute = entry.attribute;
    lines.push(
        `attr ${attribute.attribute_type()} ${attribute.unique_id()} ` +
        `${attribute.data_type()} ${attribute.num_components()} ` +
        `${attribute.normalized() ? 1 : 0} ${attribute.size()} ` +
        `${attribute.byte_stride()}`);
  }

  const face = new module.DracoInt32Array();
  for (let faceId = 0; faceId < decoded.num_faces(); ++faceId) {
    decoder.GetFaceFromMesh(decoded, faceId, face);
    lines.push(
        `face ${face.GetValue(0)} ${face.GetValue(1)} ${face.GetValue(2)}`);
  }
  module.destroy(face);

  for (const entry of entries) {
    const attribute = entry.attribute;
    const components = attribute.num_components();
    const dataType = attribute.data_type();
    const allValues =
        readAttributeValues(module, decoder, decoded, attribute);
    for (let point = 0; point < decoded.num_points(); ++point) {
      let hex = '';
      for (let component = 0; component < components; ++component) {
        hex += valueToHex(module, dataType,
                          allValues.values.GetValue(point * components +
                                                    component));
      }
      lines.push(
          `value ${attribute.attribute_type()} ${attribute.unique_id()} ` +
          `${point} ${hex}`);
    }
    allValues.destroy();
  }
  return lines.join('\n') + '\n';
}

function parseSeed(fileName) {
  const match = /^model_(\d+)\.drc$/.exec(fileName);
  if (!match) {
    return null;
  }
  return Number(match[1]);
}

async function main() {
  const modelDir = process.argv[2];
  if (!modelDir) {
    throw new Error('Usage: node draco_reference_decode.js <model-dir>');
  }
  const module = await draco3d.createDecoderModule({});
  const files = fs.readdirSync(modelDir)
                    .filter((name) => parseSeed(name) !== null)
                    .sort((left, right) => parseSeed(left) - parseSeed(right));

  for (const fileName of files) {
    const seed = parseSeed(fileName);
    const bytes = fs.readFileSync(path.join(modelDir, fileName));
    const buffer = new module.DecoderBuffer();
    buffer.Init(new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength),
                bytes.length);
    const decoder = new module.Decoder();
    const geometryType = decoder.GetEncodedGeometryType(buffer);
    if (geometryType !== module.TRIANGULAR_MESH) {
      throw new Error(`${fileName}: expected triangular mesh`);
    }
    const mesh = new module.Mesh();
    const status = decoder.DecodeBufferToMesh(buffer, mesh);
    if (!status.ok()) {
      throw new Error(`${fileName}: ${status.error_msg()}`);
    }
    process.stdout.write(dumpModel(module, decoder, mesh, seed));
    module.destroy(status);
    module.destroy(mesh);
    module.destroy(decoder);
    module.destroy(buffer);
  }
}

main().catch((error) => {
  console.error(error.stack || error);
  process.exit(1);
});
