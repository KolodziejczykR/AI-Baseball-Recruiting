import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Types for our database
export interface WaitlistEntry {
  id?: string
  email: string
  recruiting_challenge?: string
  budget?: string
  travel_team?: string
  recruiting_agency?: string
  graduation_year?: string
  desired_features?: string
  additional_info?: string
  created_at?: string
}

// Database functions
export async function insertWaitlistEntry(data: WaitlistEntry) {
  const { data: result, error } = await supabase
    .from('waitlist')
    .insert([data])
    .select()

  if (error) {
    console.error('Supabase error:', error)
    throw new Error(error.message)
  }

  return result
}

export async function checkEmailExists(email: string) {
  const { data, error } = await supabase
    .from('waitlist')
    .select('email')
    .eq('email', email)

  if (error) {
    console.error('Supabase error:', error)
    throw new Error(error.message)
  }

  return data && data.length > 0 // Returns true if email exists, false otherwise
}