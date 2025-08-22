"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"

export default function Home() {
  const router = useRouter()

  useEffect(() => {
    // Redirect to waitlist page immediately
    router.push("/waitlist")
  }, [router])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 flex items-center justify-center">
      <div className="text-center">
        <h2 className="text-6xl sm:text-5xl font-bold italic text-gray-900 leading-tight">
          The AI That {" "}
          <span className="bg-gradient-to-r from-blue-600 not-italic to-rose-600 bg-clip-text text-transparent">
            Gets You Recruited
          </span>
        </h2>
        <p className="text-gray-600 mt-2">Redirecting to waitlist...</p>
      </div>
    </div>
  )
}
