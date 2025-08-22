import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

import { insertWaitlistEntry, checkEmailExists, type WaitlistEntry } from './supabase'

export function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

// Supabase wrapper functions
export async function saveWaitlistEntry(data: WaitlistEntry) {
  try {
    const result = await insertWaitlistEntry(data)
    console.log('[SUPABASE] Waitlist entry saved:', result)
    return { data: result, error: null }
  } catch (error) {
    console.error('[SUPABASE] Error saving waitlist entry:', error)
    return { data: null, error: error instanceof Error ? error.message : 'Unknown error' }
  }
}

export async function isEmailAlreadyRegistered(email: string) {
  try {
    return await checkEmailExists(email)
  } catch (error) {
    console.error('[SUPABASE] Error checking email:', error)
    throw error
  }
}