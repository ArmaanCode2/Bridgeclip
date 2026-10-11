import { AssistantToolError } from './tool-types'

// Validate tool arguments against the small JSON Schema subset the tool
// definitions use, so every tool sees well-typed input and the agent gets a
// precise message it can correct instead of a failure deep in a handler.

type Schema = Record<string, unknown>

function describeType(value: unknown): string {
  if (Array.isArray(value)) return 'array'
  if (value === null) return 'null'
  return typeof value
}

function check(value: unknown, schema: Schema, path: string): void {
  const enumValues = schema.enum
  if (Array.isArray(enumValues) && !enumValues.includes(value)) {
    throw new AssistantToolError(`${path} must be one of: ${enumValues.map((entry) => JSON.stringify(entry)).join(', ')}`)
  }
  switch (schema.type) {
    case 'string': {
      if (typeof value !== 'string') throw new AssistantToolError(`${path} must be a string, got ${describeType(value)}`)
      if (typeof schema.minLength === 'number' && value.length < schema.minLength) throw new AssistantToolError(`${path} must not be empty`)
      if (typeof schema.maxLength === 'number' && value.length > schema.maxLength) throw new AssistantToolError(`${path} must be at most ${schema.maxLength} characters`)
      if (typeof schema.pattern === 'string' && !new RegExp(schema.pattern).test(value)) throw new AssistantToolError(`${path} has an invalid format`)
      return
    }
    case 'integer':
    case 'number': {
      if (typeof value !== 'number' || !Number.isFinite(value)) throw new AssistantToolError(`${path} must be a number`)
      if (schema.type === 'integer' && !Number.isInteger(value)) throw new AssistantToolError(`${path} must be a whole number`)
      if (typeof schema.minimum === 'number' && value < schema.minimum) throw new AssistantToolError(`${path} must be at least ${schema.minimum}`)
      if (typeof schema.maximum === 'number' && value > schema.maximum) throw new AssistantToolError(`${path} must be at most ${schema.maximum}`)
      return
    }
    case 'boolean':
      if (typeof value !== 'boolean') throw new AssistantToolError(`${path} must be true or false`)
      return
    case 'array': {
      if (!Array.isArray(value)) throw new AssistantToolError(`${path} must be an array`)
      if (typeof schema.minItems === 'number' && value.length < schema.minItems) throw new AssistantToolError(`${path} needs at least ${schema.minItems} item(s)`)
      if (typeof schema.maxItems === 'number' && value.length > schema.maxItems) throw new AssistantToolError(`${path} allows at most ${schema.maxItems} items`)
      const items = schema.items as Schema | undefined
      if (items) value.forEach((item, index) => check(item, items, `${path}[${index}]`))
      return
    }
    case 'object': {
      if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new AssistantToolError(`${path} must be an object`)
      const properties = (schema.properties ?? {}) as Record<string, Schema>
      const required = Array.isArray(schema.required) ? schema.required as string[] : []
      for (const key of required) {
        if ((value as Record<string, unknown>)[key] === undefined) throw new AssistantToolError(`${path === 'input' ? '' : `${path}.`}${key} is required`)
      }
      for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
        const property = properties[key]
        if (!property) {
          if (schema.additionalProperties === false) throw new AssistantToolError(`Unknown field ${path === 'input' ? '' : `${path}.`}${key}`)
          continue
        }
        if (entry === undefined || (entry === null && !required.includes(key))) continue
        check(entry, property, path === 'input' ? key : `${path}.${key}`)
      }
      return
    }
    default:
      return
  }
}

/** Throws AssistantToolError with a precise message when input doesn't match. */
export function validateToolInput(input: unknown, schema: Schema): Record<string, unknown> {
  check(input ?? {}, schema, 'input')
  // Optional fields sent as null mean "not provided".
  return Object.fromEntries(Object.entries((input ?? {}) as Record<string, unknown>).filter(([, value]) => value !== null && value !== undefined))
}
