export default function SuccessPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 flex items-center justify-center">
      <div className="max-w-2xl mx-auto px-4 text-center">
        <div className="bg-white rounded-3xl p-12 shadow-xl border border-gray-100">
          <div className="w-20 h-20 bg-gradient-to-r from-green-500 to-green-600 rounded-full flex items-center justify-center mb-8 mx-auto">
            <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
          
          <h1 className="text-4xl font-bold text-gray-900 mb-6">
            Perfect! You&apos;re all set!
          </h1>
          
          <p className="text-xl text-gray-600 mb-8 leading-relaxed">
            Thanks for joining the BaseballPATH waitlist and sharing your feedback. We&apos;ll be in touch soon with more details!
          </p>

          <div className="bg-gradient-to-r from-blue-50 to-red-50 rounded-xl p-6 border border-blue-100">
            <p className="text-gray-700 font-medium">
              Keep an eye on your inbox - we&apos;ll send you exclusive updates and launch information as soon as it&apos;s ready!
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}